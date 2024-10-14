import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from scipy.stats import mode
import numpy as np
import cv2
import os
import sys
import torch
sys.path.append('./offseg')

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory


class SuperpixelGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Semantic Annotator GUI")

        self.image_folder = ""
        self.image_paths = []
        self.current_image_index = 0
        self.image_name = None
        self.image = None
        self.superpixels = None
        self.prompt_mask = None
        self.save_name = ""
        self.alpha = 0.2
        self.combined_masks = None
        self.polygon_mask = None
        self.mask = None
        self.input_points = []
        self.input_labels = []
        self.polygon_points = []
        self.rect_mask = None
        self.label_prev = None
        self.image_prev = None
        self.rec_prev = None

        self.start_x = None
        self.start_y = None
        self.adj_point = None

        self.sam = sam_model_registry["vit_h"](checkpoint = './models/sam_vit_h_4b8939.pth')
        self.sam.to("cuda:0")

        self.offseg_cfg = cfg_factory["bisenetv2"]
        self.offseg_net = model_factory[self.offseg_cfg.model_type](4)
        self.offseg_net.load_state_dict(torch.load('./models/model_final3.pth'))
        self.offseg_net.eval()
        self.offseg_net.to("cuda:0")
        self.to_tensor = T.ToTensor(mean=(0.3257, 0.3690, 0.3223), std=(0.2112, 0.2148, 0.2115))
        
        self.open_button = tk.Button(self.master, text="Open Folder", command=self.open_folder)
        self.open_button.pack()

        self.save_button = tk.Button(self.master, text="Save", command=self.save_images, state=tk.DISABLED)
        self.save_button.place(x = 800, y = 20)

        self.view_image_button = tk.Button(self.master, text="Raw Image")
        self.view_image_button.place(x = 200, y = 20)
        self.view_image_button.bind("<ButtonPress-1>", lambda event: self.show_image_cus(self.org_image))
        self.view_image_button.bind("<ButtonRelease-1>", lambda event: self.show_image_custom())

        self.compactness_var = tk.IntVar()
        self.compactness_scale = tk.Scale(self.master, from_=1, to=100, length=300, orient=tk.HORIZONTAL, label="Compactness", variable=self.compactness_var)
        self.compactness_scale.set(25)
        self.compactness_scale.pack()

        self.sam_var = tk.IntVar()
        self.sam_scale = tk.Scale(self.master, from_=5, to=50, length=300, orient=tk.HORIZONTAL, label="SAM_labels", variable=self.sam_var)
        self.sam_scale.set(25)
        self.sam_scale.pack()

        self.num_superpixels_var = tk.IntVar()
        self.num_superpixels_scale = tk.Scale(self.master, from_=5, to=200, length=300, orient=tk.HORIZONTAL, label="Number of Superpixels", variable=self.num_superpixels_var)
        self.num_superpixels_scale.set(100)
        self.num_superpixels_scale.pack()

        self.generate_button = tk.Button(self.master, text="Generate Superpixels", command=self.generate_superpixels)
        self.generate_button.place(x = 150, y = 200)

        self.sam_generate_button = tk.Button(self.master, text="Generate SAM masks", command=self.generate_masks)
        self.sam_generate_button.pack()

        self.offseg_seg_button = tk.Button(self.master, text="Segment using offseg", command=self.segment_offseg)
        self.offseg_seg_button.place(x = 150, y = 100)

        self.revert_button = tk.Button(self.master, text="Revert previous selection", command=self.revert_superpixels)
        self.revert_button.place(x = 800, y = 200)

        self.view_semantics_button = tk.Button(self.master, text="View Semantics", command=self.view_semantics)
        self.view_semantics_button.place(x = 980, y = 250)

        self.interpolate_button = tk.Button(self.master, text="Interpolate", command=self.interpolate)
        self.interpolate_button.place(x = 980, y = 300)

        self.num_segments_label = tk.Label(self.master, text="Actual number of superpixels: -")
        self.num_segments_label.pack()

        self.num_masks_label = tk.Label(self.master, text="Actual number of SAM masks: -")
        self.num_masks_label.pack()

        self.segment_rectangle_button = tk.Button(self.master, text = "Segment Rectangle", command = self.segment_rectangle)
        self.segment_rectangle_button.place(x = 800, y = 150)

        self.percentage_label = tk.Label(self.master, text="Percentage of superpixels labeled: -")
        self.percentage_label.pack()

        self.original_image = None  # New instance variable

        self.color_labels = {
            0: ("void", "#000000"), 1: ("dirt", "#6c4014"), 3: ("grass", "#006600"), 4: ("trees", "#00ff00"),
            5: ("pole", "#009999"), 6: ("water", "#0080ff"), 7: ("sky", "#0000ff"), 8: ("vehicle", "#ffff00"),
            9: ("object", "#ff007f"), 10: ("asphalt", "#404040"), 12: ("build", "#ff0000"), 15: ("log", "#660000"),
            17: ("person", "#cc99ff"), 18: ("fence", "#6600cc"), 19: ("bush", "#ff99cc"), 23: ("concrete", "#aaaaaa"),
            27: ("barrier", "#2979FF"), 31: ("puddle", "#86ffef"), 33: ("mud", "#634222"), 34: ('rubble','#6e168a'),
            35: ("mulch", "#8000ff"), 36: ("gravel", "#808080")
        }

        self.color_buttons = []
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.RIGHT)
        label_id_def = 0
        for label_id, (label_name, label_color) in self.color_labels.items():
            color_button = tk.Button(button_frame, text=label_name, bg=label_color,
                                    command=lambda id=label_id: self.select_label(id))
            row = label_id_def // 2
            col = label_id_def % 2
            color_button.grid(row=row, column=col, padx=5, pady=5)
            self.color_buttons.append(color_button)
            label_id_def += 1

        self.canvas = tk.Canvas(self.master, width=960, height=720)
        self.canvas.pack()

        self.selected_label = None
        self.selected_color = None

        self.segment_labels = {}
        self.canvas.bind("<Button-1>", self.prompt_masks)  # Bind the label assignment function to the left button click

        self.interact_button = tk.Button(self.master, text="Interact with SAM", command=self.interact)
        self.interact_button.place(x = 150, y = 150)

        self.rectrangle_button = tk.Button(self.master, text="Create Rectangle", command=self.draw_rectangle)
        self.rectrangle_button.place(x = 800, y = 100)
        
        self.polygon_button = tk.Button(self.master, text="Create Polygon", command=self.draw_polygon)
        self.polygon_button.place(x = 950, y = 100)

        self.adjust_button = tk.Button(self.master, text="Adjust Polygon", command=self.adjust_polygon_dir)
        self.adjust_button.place(x = 950, y = 150)

        

    def open_folder(self):
        self.image_folder = filedialog.askdirectory()
        if self.image_folder:
            self.image_paths = sorted([os.path.join(self.image_folder, filename) for filename in os.listdir(self.image_folder) if
                                filename.endswith(('.jpg', '.png'))])
            if self.image_paths:
                self.load_image(0)

    def load_image(self, index):
        if 0 <= index < len(self.image_paths):
            image_path = self.image_paths[index]
            self.image_name = image_path
            self.original_image = Image.open(image_path)  # Store the original opened image
            self.org_np = np.asarray(self.original_image)
            self.org_image = self.original_image.resize((960, 720))
            self.image = self.original_image.resize((960, 720))  # Resize the image to fit in the canvas
            self.np_image = np.asarray(self.image)
            self.label_hist = np.zeros((np.asarray(self.image).shape[0], np.asarray(self.image).shape[1]))
            self.show_image()
            self.current_image_index = index

            self.save_button.configure(state=tk.DISABLED)

    def draw_rectangle(self):
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        if not isinstance(self.rec_prev, type(None)):
            self.canvas.delete(self.rec_prev)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def on_mouse_drag(self, event):
        pass
        # expand rectangle as you drag the mouse
        #self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)    

    def on_button_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        self.rec_prev = self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline='red')
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        pass

    def show_image(self):
        if self.image:
            tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image
    
    def interact(self):
        self.canvas.bind("<Button-1>", self.prompt_masks)
        tk_image = ImageTk.PhotoImage(self.org_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image
        self.input_points = []
        self.input_labels = []
        pass

    def draw_polygon(self):
        self.input_points = []
        self.canvas.bind("<Button-1>", self.create_polygon)
        self.canvas.bind("<Button-3>", self.close_polygon)
        tk_image = ImageTk.PhotoImage(self.org_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image
        pass

    def adjust_polygon_dir(self):
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<Button-3>")
        self.canvas.bind("<Button-1>", self.on_button_press_poly)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag_poly)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release_poly)
        pass

    def on_button_press_poly(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        nodes = np.asarray(self.polygon_points)
        dist_2 = np.sum((nodes - np.array([self.start_x,self.start_y]))**2, axis=1)
        self.adj_point =  np.argmin(dist_2)

    def on_mouse_drag_poly(self, event):
        pass

    def on_button_release_poly(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        self.polygon_points[self.adj_point] = [int(self.end_x),int(self.end_y)]
        self.canvas.delete("all")
        tk_image = ImageTk.PhotoImage(self.org_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image = tk_image)
        self.canvas.image = tk_image
        self.canvas.create_polygon(self.polygon_points, outline='red', fill = '')
        for point in self.polygon_points:
            self.canvas.create_oval(point[0]-2,point[1]-2,point[0]+2,point[1]+2,fill = 'black')
        pass        

    def show_image_cus(self,image):
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image
        
    def show_image_custom(self):
        if not isinstance(self.superpixels, type(None)):
            self.show_superpixels()
        else:
            self.show_masks()
    
    def view_semantics(self):
        marked_image = mark_boundaries(np.array(self.image), self.label_hist.astype(np.uint8))
        self.marked_image = (marked_image * 255).astype(np.uint8)
        out_img = Image.fromarray(self.marked_image)

        tk_image = ImageTk.PhotoImage(out_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image
        
    def revert_superpixels(self):
        self.label_hist = self.label_prev
        self.image = self.image_prev
        if not isinstance(self.superpixels, type(None)):
            self.show_superpixels()
        elif not isinstance(self.combined_masks, type(None)):
            self.show_masks()
        elif not isinstance(self.prompt_mask, type(None)):
            self.show_prompt_masks(False)
        elif not isinstance(self.offseg_mask, type(None)):
            self.show_offseg_masks()
        elif not isinstance(self.rect_mask, type(None)):
            self.show_rect_masks()
        elif not isinstance(self.polygon_mask, type(None)):
            self.show_poly_masks()

    def generate_superpixels(self):
        compactness = self.compactness_var.get()
        num_segments = self.num_superpixels_var.get()
        self.superpixels = slic(np.array(self.np_image), n_segments=num_segments, compactness=compactness)

        self.show_superpixels()
        actual_segments = np.unique(self.superpixels)
        self.num_segments_label.configure(text="Actual number of superpixels: " + str(len(actual_segments)))
        self.canvas.bind("<Button-1>", self.assign_label)
        self.combined_masks = None
        self.prompt_mask = None
        self.offseg_mask = None
        self.rect_mask = None
        self.polygon_mask = None

    def segment_rectangle(self):
        self.prompt_generator = SamPredictor(self.sam)
        self.prompt_generator.set_image(np.array(self.org_image))
        startx = int(self.start_x)  if int(self.start_x) < int(self.end_x) else int(self.end_x)
        starty = int(self.start_y) if int(self.start_y) < int(self.end_y) else int(self.end_y)
        endx = int(self.end_x) if int(self.start_x) < int(self.end_x) else int(self.start_x)
        endy = int(self.end_y) if int(self.start_y) < int(self.end_y) else int(self.start_y)
        self.input_box = np.array([startx,starty,endx,endy])
        masks,scores,logits = self.prompt_generator.predict(point_coords = None, point_labels = None, box = self.input_box[None,:],multimask_output = False)
        nth_mask = np.argmax(scores)
        segment_mask = masks[nth_mask]*1
        tmp_mask = np.ones((masks[0].shape[0],masks[0].shape[1]), dtype=np.uint8)*2
        tmp_mask[starty:endy,startx:endx] = segment_mask[starty:endy,startx:endx]
        self.rect_mask = tmp_mask
        self.show_rect_masks()
        self.combined_masks = None
        self.superpixels = None
        self.offseg_mask = None
        self.prompt_mask = None
        self.polygon_mask = None

    def create_polygon(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.polygon_points.append([int(x),int(y)])
        self.canvas.create_oval(x-2,y-2,x+2,y+2,fill = 'black')
        if len(self.polygon_points) > 1:
            self.canvas.create_line(self.polygon_points[-2][0],self.polygon_points[-2][1],self.polygon_points[-1][0],self.polygon_points[-1][1],fill = 'red')
        
    def close_polygon(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.polygon_points.append([int(x),int(y)])
        self.canvas.create_oval(x-2,y-2,x+2,y+2,fill = 'black')
        self.canvas.create_line(self.polygon_points[-1][0],self.polygon_points[-1][1],self.polygon_points[-2][0],self.polygon_points[-2][1],fill = 'red')
        self.canvas.create_line(self.polygon_points[-1][0],self.polygon_points[-1][1],self.polygon_points[0][0],self.polygon_points[0][1],fill = 'red')
        self.polygon_mask = np.zeros((self.np_image.shape[0],self.np_image.shape[1]),dtype = np.uint8)
        self.polygon_mask = cv2.fillPoly(self.polygon_mask,[np.array(self.polygon_points)],1)
        self.show_poly_masks()
        self.combined_masks = None
        self.superpixels = None
        self.offseg_mask = None
        self.rect_mask = None
        self.prompt_mask = None

    def segment_offseg(self):
        tensor_image = self.to_tensor(dict(im=np.array(self.np_image), lb=None))['im'].unsqueeze(0).cuda()
        self.offseg_mask = self.offseg_net(tensor_image)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        self.show_offseg_masks()
        self.combined_masks = None
        self.superpixels = None
        self.prompt_mask = None
        self.rect_mask = None
        self.polygon_mask = None

    def prompt_masks(self,event):
        self.prompt_generator = SamPredictor(self.sam)
        self.prompt_generator.set_image(np.array(self.org_image))

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.input_points.append([int(x),int(y)])
        self.input_labels.append(1)
        self.ppt_masks, score, logits = self.prompt_generator.predict(point_coords = np.asarray(self.input_points), point_labels = np.asarray(self.input_labels),multimask_output = False)
        nth_mask = np.argmax(score)
        self.prompt_mask = self.ppt_masks[nth_mask]*1
        self.show_prompt_masks(True)
        self.combined_masks = None
        self.superpixels = None
        self.offseg_mask = None
        self.rect_mask = None
        self.polygon_mask = None

    def generate_masks(self):
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side = self.sam_var.get(),box_nms_thresh = 0.1,pred_iou_thresh= 0.6) ## points_per_side adjusts the number of masks generated
        masks = self.mask_generator.generate(np.asarray(self.np_image))
        self.segmentation_masks = [masks[i]["segmentation"]*(i+1) for i in range(len(masks))]
        self.segmentation_masks = np.array(self.segmentation_masks)
        self.num_masks_label.configure(text="Actual number of SAM masks: " + str(len(self.segmentation_masks)))
        self.canvas.bind("<Button-1>", self.assign_label)
        self.combined_masks = np.sum(self.segmentation_masks,axis = 0,dtype = np.uint8)
        self.show_masks()
        self.superpixels = None
        self.prompt_mask = None
        self.offseg_mask = None
        self.rect_mask = None
        self.polygon_mask = None

    def show_prompt_masks(self,show_raw):
        if self.prompt_mask is not None and show_raw:
            marked_image = mark_boundaries(np.array(self.np_image), self.prompt_mask)
            self.marked_image = (marked_image * 255).astype(np.uint8)
            out_img = Image.fromarray(self.marked_image)

            tk_image = ImageTk.PhotoImage(out_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image
        else:
            marked_image = mark_boundaries(np.array(self.image), self.prompt_mask)
            self.marked_image = (marked_image * 255).astype(np.uint8)
            out_img = Image.fromarray(self.marked_image)

            tk_image = ImageTk.PhotoImage(out_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image
        
    def show_offseg_masks(self):
         if self.offseg_mask is not None:
                marked_image = mark_boundaries(np.array(self.image), self.offseg_mask)
                self.marked_image = (marked_image * 255).astype(np.uint8)
                out_img = Image.fromarray(self.marked_image)
    
                tk_image = ImageTk.PhotoImage(out_img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
                self.canvas.image = tk_image

    def show_masks(self):
       if self.combined_masks is not None:
            marked_image = mark_boundaries(np.array(self.image), self.combined_masks)
            self.marked_image = (marked_image * 255).astype(np.uint8)
            out_img = Image.fromarray(self.marked_image)

            tk_image = ImageTk.PhotoImage(out_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image

    def show_rect_masks(self):
        if self.rect_mask is not None:
            marked_image = mark_boundaries(np.array(self.image), self.rect_mask)
            self.marked_image = (marked_image * 255).astype(np.uint8)
            out_img = Image.fromarray(self.marked_image)

            tk_image = ImageTk.PhotoImage(out_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image

    def show_poly_masks(self):
        if self.polygon_mask is not None:
            marked_image = mark_boundaries(np.array(self.image), self.polygon_mask.astype(np.uint8))
            self.marked_image = (marked_image * 255).astype(np.uint8)
            out_img = Image.fromarray(self.marked_image)

            tk_image = ImageTk.PhotoImage(out_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image
                
    def show_superpixels(self):
        if self.image and self.superpixels is not None:
            segmented_image = mark_boundaries(np.array(self.image), self.superpixels)
            segmented_image = (segmented_image * 255).astype(np.uint8)
            self.segmented_image = Image.fromarray(segmented_image)

            tk_image = ImageTk.PhotoImage(self.segmented_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image

    def select_label(self, label_id):
        self.selected_label = label_id
        self.selected_color = self.color_labels[label_id][1]
        self.canvas.bind("<Button-1>", self.assign_label)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def assign_label(self, event):
        if self.selected_label is None or not self.image:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        self.image_prev = self.image.copy()
        self.label_prev = self.label_hist.copy()

        if not isinstance(self.superpixels, type(None)):
            segment_id = self.superpixels[int(y), int(x)]
            # Segment the superpixel using the color of the selected label
            label_color = self.selected_color
            self.mask = (self.superpixels == segment_id)
            segmented_image = np.array(self.image)
            segmented_image[self.mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            self.label_hist[self.mask] = self.selected_label
            #print(self.label_hist)
            self.image = Image.fromarray(segmented_image)

            # Assign the label ID to the segment ID
            self.segment_labels[segment_id] = self.selected_label

            # Update the displayed superpixel image with the segmented label
            self.show_superpixels()

            return
        if not isinstance(self.combined_masks, type(None)):
            segment_id = self.combined_masks[int(y), int(x)]
            # Segment the superpixel using the color of the selected label
            label_color = self.selected_color
            self.mask = (self.combined_masks == segment_id)
            segmented_image = np.array(self.image)
            segmented_image[self.mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            self.label_hist[self.mask] = self.selected_label
            #print(self.label_hist)
            self.image = Image.fromarray(segmented_image)

            # Assign the label ID to the segment ID
            self.segment_labels[segment_id] = self.selected_label

            # Update the displayed superpixel image with the segmented label
            self.show_masks()
        
        if not isinstance(self.prompt_mask, type(None)):
            segment_id = self.prompt_mask[int(y), int(x)]
            # Segment the superpixel using the color of the selected label
            label_color = self.selected_color
            self.mask = (self.prompt_mask == segment_id)
            segmented_image = np.array(self.image)
            segmented_image[self.mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            self.label_hist[self.mask] = self.selected_label
            #print(self.label_hist)
            self.image = Image.fromarray(segmented_image)

            # Assign the label ID to the segment ID
            self.segment_labels[segment_id] = self.selected_label

            # Update the displayed superpixel image with the segmented label
            self.show_prompt_masks(False)

        if not isinstance(self.offseg_mask, type(None)):
            segment_id = self.offseg_mask[int(y), int(x)]
            # Segment the superpixel using the color of the selected label
            label_color = self.selected_color
            self.mask = (self.offseg_mask == segment_id)
            segmented_image = np.array(self.image)
            segmented_image[self.mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            self.label_hist[self.mask] = self.selected_label
            #print(self.label_hist)
            self.image = Image.fromarray(segmented_image)

            # Assign the label ID to the segment ID
            self.segment_labels[segment_id] = self.selected_label

            # Update the displayed superpixel image with the segmented label
            self.show_offseg_masks()
        if not isinstance(self.rect_mask, type(None)):
            segment_id = self.rect_mask[int(y), int(x)]
            # Segment the superpixel using the color of the selected label
            label_color = self.selected_color
            self.mask = (self.rect_mask == segment_id)
            segmented_image = np.array(self.image)
            segmented_image[self.mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            self.label_hist[self.mask] = self.selected_label
            #print(self.label_hist)
            self.image = Image.fromarray(segmented_image)

            # Assign the label ID to the segment ID
            self.segment_labels[segment_id] = self.selected_label

            # Update the displayed superpixel image with the segmented label
            self.show_rect_masks()
        if not isinstance(self.polygon_mask, type(None)):
            segment_id = self.polygon_mask[int(y), int(x)]
            # Segment the superpixel using the color of the selected label
            label_color = self.selected_color
            self.mask = (self.polygon_mask == segment_id)
            segmented_image = np.array(self.image)
            segmented_image[self.mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
            self.label_hist[self.mask] = self.selected_label
            #print(self.label_hist)
            self.image = Image.fromarray(segmented_image)

            # Assign the label ID to the segment ID
            self.segment_labels[segment_id] = self.selected_label
            self.polygon_points = []

            # Update the displayed superpixel image with the segmented label
            self.show_poly_masks()
        #print(np.count_nonzero(self.label_hist==0))
        percentage = np.count_nonzero(self.label_hist.flatten()==0)/len(self.label_hist.flatten())*3*100
        self.percentage_label.configure(text="Percentage of PIXELS labeled: " + str(100-percentage))
        # Check if all superpixels have been labeled
        if len(self.segment_labels) == len(np.unique(self.superpixels)):
           self.save_button.configure(state=tk.NORMAL)

        if np.any(self.label_hist == 0):
            self.save_button.configure(state=tk.DISABLED)
        else:
            self.save_button.configure(state=tk.NORMAL)
    
    def interpolate(self):
        kernel_size = 7
        arr = self.label_hist.copy()
        rows, cols = arr.shape
        pad_size = kernel_size // 2  # Calculate the padding size

        # Pad the array with zeros to handle edge cases
        padded_arr = np.pad(self.label_hist, pad_size, mode='constant', constant_values=0)

        for i in range(rows):
            for j in range(cols):
                if arr[i, j] == 0:
                    # Extract the kernel centered at the current element
                    kernel = padded_arr[i:i+kernel_size, j:j+kernel_size]
                    
                    # Find the most frequent non-zero value in the kernel
                    non_zero_elements = kernel[kernel != 0]
                    if non_zero_elements.size > 0:
                        most_frequent = mode(non_zero_elements, axis=None).mode[0]
                        # Replace the zero with the most frequent non-zero value found
                        arr[i, j] = most_frequent
        self.label_hist = arr
        percentage = np.count_nonzero(self.label_hist.flatten()==0)/len(self.label_hist.flatten())*100
        self.percentage_label.configure(text="Percentage of PIXELS labeled: " + str(100-percentage))

        col_image = self.label_image(False)
        tk_image = ImageTk.PhotoImage(col_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image


    def save_images(self):
        save_folder = "images"
        os.makedirs(save_folder, exist_ok=True)
        original_save_path = os.path.join(save_folder, os.path.basename(self.image_name))
        self.original_image.save(original_save_path)  # Save the original image

        color_labels_folder = "color_labels"
        os.makedirs(color_labels_folder, exist_ok=True)
        color_save_path = os.path.join(color_labels_folder, os.path.basename(self.image_name))
        labeled_image = self.label_image()
        labeled_image.save(color_save_path)

        labels_folder = "labels"
        os.makedirs(labels_folder, exist_ok=True)
        gray_save_path = os.path.join(labels_folder, os.path.basename(self.image_name))
        #gray_array = self.get_gray_array()
        cv2.imwrite(gray_save_path, cv2.resize(self.label_hist, (self.org_np.shape[1],self.org_np.shape[0]), interpolation=cv2.INTER_NEAREST))

        #removing the image from work folder
        os.remove(self.image_name)

        self.segment_labels = {}  # Clear the segment labels for the next image
        self.current_image_index += 1
        self.load_image(self.current_image_index)  # Load the next image

    def label_image(self, event = True):
        labeled_image = np.array(self.image)
        for segment_id, label_id in self.segment_labels.items():
            mask = (self.label_hist == label_id)
            label_color = self.color_labels[label_id][1]
            labeled_image[mask] = tuple(int(label_color[i:i + 2], 16) for i in (1, 3, 5))
        if event:
            labeled_image = cv2.resize(labeled_image, (self.org_np.shape[1],self.org_np.shape[0]),interpolation=cv2.INTER_NEAREST)
        else:
            labeled_image = cv2.resize(labeled_image, (960,720),interpolation=cv2.INTER_NEAREST)
        return Image.fromarray(labeled_image)

    def get_gray_array(self):
        image_shape = self.image.size
        gray_array = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)
        for segment_id, label_id in self.segment_labels.items():
            gray_array[self.superpixels == segment_id] = label_id

        return gray_array

    def next_image(self):
        self.segment_labels = {}  # Clear the segment labels for the next image
        self.current_image_index += 1
        self.load_image(self.current_image_index)  # Load the next image


if __name__ == "__main__":
    root = tk.Tk()
    gui = SuperpixelGUI(root)
    root.mainloop()