'''
Code for BatchedNMS

Change the path to your ONNX model without NMS

Change the num_classes on your count of classes

Change in origin_output 'node.name == Concat_NUMBER[0]'. 
You can see the last Concat number if you open your model on www.netron.app (https://github.com/lutzroeder/netron)

Check out README.md, I show you how to view it clearly there.
'''

import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("./best.onnx")) # change on your path

num_classes = 11 # change on your number

# input
origin_output = [node for node in graph.nodes if node.name == "Concat_386"][0] # change 386 on your last Concat number
print(origin_output.outputs)

starts_wh = gs.Constant("starts_wh", values=np.array([0, 0, 0], dtype=np.int64))
ends_wh = gs.Constant("ends_wh", values=np.array([1, 25200, 4], dtype=np.int64))

starts_object = gs.Constant("starts_object",
                            values=np.array([0, 0, 4], dtype=np.int64))  # in standard Yolov7 is data name:615
ends_object = gs.Constant("ends_object", values=np.array([1, 25200, 5], dtype=np.int64))

starts_conf = gs.Constant("starts_conf", values=np.array([0, 0, 5], dtype=np.int64))
ends_conf = gs.Constant("ends_conf", values=np.array([1, 25200, num_classes + 5], dtype=np.int64))

# output
# for boxes
box_xywh_0 = gs.Variable(name="box_xywh_0", shape=(1, 25200, 4), dtype=np.float32)

# for scores
object_prob_0 = gs.Variable(name="object_prob_0", shape=(1, 25200, 1), dtype=np.float32)
label_conf_0 = gs.Variable(name='label_conf_0', shape=(1, 25200, num_classes), dtype=np.float32)

# slice
box_xywh_node = gs.Node(op="Slice", inputs=[origin_output.outputs[0], starts_wh, ends_wh], outputs=[box_xywh_0])
box_prob_node = gs.Node(op="Slice", inputs=[origin_output.outputs[0], starts_object, ends_object],
                        outputs=[object_prob_0])
box_conf_node = gs.Node(op="Slice", inputs=[origin_output.outputs[0], starts_conf, ends_conf], outputs=[label_conf_0])

# identity
box_xywh = gs.Variable(name="box_xywh", shape=(1, 25200, 4), dtype=np.float32)
object_prob = gs.Variable(name="object_prob", shape=(1, 25200, 1), dtype=np.float32)
label_conf = gs.Variable(name='label_conf', shape=(1, 25200, num_classes), dtype=np.float32)

identity_node_wh = gs.Node(op="Identity", inputs=[box_xywh_0], outputs=[box_xywh])
identity_node_prob = gs.Node(op="Identity", inputs=[object_prob_0], outputs=[object_prob])
identity_node_conf = gs.Node(op="Identity", inputs=[label_conf_0], outputs=[label_conf])

print(identity_node_wh)

# input
starts_1 = gs.Constant("starts_x", values=np.array([0, 0, 0], dtype=np.int64))
ends_1 = gs.Constant("ends_x", values=np.array([1, 25200, 1], dtype=np.int64))

starts_2 = gs.Constant("starts_y", values=np.array([0, 0, 1], dtype=np.int64))
ends_2 = gs.Constant("ends_y", values=np.array([1, 25200, 2], dtype=np.int64))

starts_3 = gs.Constant("starts_w", values=np.array([0, 0, 2], dtype=np.int64))
ends_3 = gs.Constant("ends_w", values=np.array([1, 25200, 3], dtype=np.int64))

starts_4 = gs.Constant("starts_h", values=np.array([0, 0, 3], dtype=np.int64))
ends_4 = gs.Constant("ends_h", values=np.array([1, 25200, 4], dtype=np.int64))

# output
x = gs.Variable(name="x_center", shape=(1, 25200, 1), dtype=np.float32)
y = gs.Variable(name="y_center", shape=(1, 25200, 1), dtype=np.float32)
w = gs.Variable(name="w", shape=(1, 25200, 1), dtype=np.float32)
h = gs.Variable(name="h", shape=(1, 25200, 1), dtype=np.float32)

# xywh_split_node = gs.Node(op="Split",inputs=[box_xywh],outputs= [x,y,w,h] )
x_node = gs.Node(op="Slice", inputs=[box_xywh, starts_1, ends_1], outputs=[x])
y_node = gs.Node(op="Slice", inputs=[box_xywh, starts_2, ends_2], outputs=[y])
w_node = gs.Node(op="Slice", inputs=[box_xywh, starts_3, ends_3], outputs=[w])
h_node = gs.Node(op="Slice", inputs=[box_xywh, starts_4, ends_4], outputs=[h])

# input
div_val = gs.Constant("div_val", values=np.array([2], dtype=np.float32))
div_val_ = gs.Constant("div_val_", values=np.array([-2], dtype=np.float32))
# output
w_ = gs.Variable(name="w_half_", shape=(1, 25200, 1), dtype=np.float32)
wplus = gs.Variable(name="w_half_plus", shape=(1, 25200, 1), dtype=np.float32)
h_ = gs.Variable(name="h_half_", shape=(1, 25200, 1), dtype=np.float32)
hplus = gs.Variable(name="h_half_plus", shape=(1, 25200, 1), dtype=np.float32)

w_node_ = gs.Node(op="Div", inputs=[w, div_val_], outputs=[w_])
w_node_plus = gs.Node(op="Div", inputs=[w, div_val], outputs=[wplus])
h_node_ = gs.Node(op="Div", inputs=[h, div_val_], outputs=[h_])
h_node_plus = gs.Node(op="Div", inputs=[h, div_val], outputs=[hplus])

# output
x1 = gs.Variable(name="x1", shape=(1, 25200, 1), dtype=np.float32)
y1 = gs.Variable(name="y1", shape=(1, 25200, 1), dtype=np.float32)
x2 = gs.Variable(name="x2", shape=(1, 25200, 1), dtype=np.float32)
y2 = gs.Variable(name="y2", shape=(1, 25200, 1), dtype=np.float32)

x1_node = gs.Node(op="Add", inputs=[x, w_], outputs=[x1])
x2_node = gs.Node(op="Add", inputs=[x, wplus], outputs=[x2])
y1_node = gs.Node(op="Add", inputs=[y, h_], outputs=[y1])
y2_node = gs.Node(op="Add", inputs=[y, hplus], outputs=[y2])

# concat
# output

boxes_0 = gs.Variable(name="boxes_0", shape=(1, 25200, 4), dtype=np.float32)

boxes_node_0 = gs.Node(op="Concat", inputs=[x1, y1, x2, y2], outputs=[boxes_0], attrs={"axis": 2})

shapes = gs.Constant("shape", values=np.array([1, 25200, 1, 4], dtype=np.int64))

# output
boxes = gs.Variable(name="boxes", shape=(1, 25200, 1, 4), dtype=np.float32)

boxes_node = gs.Node(op="Reshape", inputs=[boxes_0, shapes], outputs=[boxes])

# prob
scores = gs.Variable(name="scores", shape=(1, 25200, num_classes), dtype=np.float32)

# Mul Node
scores_node = gs.Node(op="Mul", inputs=[label_conf, object_prob], outputs=[scores])

graph.nodes.extend(
    [box_xywh_node, box_prob_node, box_conf_node, identity_node_wh, identity_node_prob, identity_node_conf,
     x_node, y_node, w_node, h_node,
     w_node_, w_node_plus, h_node_, h_node_plus, x1_node, x2_node, y1_node, y2_node, boxes_node_0, boxes_node,
     scores_node])

graph.outputs = [boxes, scores]

graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "./postprocessed_model.onnx")
