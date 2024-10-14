### Annotating Custom Dataset.

#### Inorder to annotate your custom dataset, you would need to change a few parameters in [gui.py](./gui.py)

Modify your labels, label id and color for your custom dataset at Line:118 of [gui.py](./gui.py)

```
self.color_labels = {
            0: ("void", "#000000"), 1: ("dirt", "#6c4014"), 3: ("grass", "#006600"), 4: ("trees", "#00ff00"),
            5: ("pole", "#009999"), 6: ("water", "#0080ff"), 7: ("sky", "#0000ff"), 8: ("vehicle", "#ffff00"),
            9: ("object", "#ff007f"), 10: ("asphalt", "#404040"), 12: ("build", "#ff0000"), 15: ("log", "#660000"),
            17: ("person", "#cc99ff"), 18: ("fence", "#6600cc"), 19: ("bush", "#ff99cc"), 23: ("concrete", "#aaaaaa"),
            27: ("barrier", "#2979FF"), 31: ("puddle", "#86ffef"), 33: ("mud", "#634222"), 34: ('rubble','#6e168a'),
            35: ("mulch", "#8000ff"), 36: ("gravel", "#808080")
        }
```
