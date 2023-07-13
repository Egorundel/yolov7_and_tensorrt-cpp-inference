'''
BatchedNMS Plugin 

Change number_classes on your count of classes

'''
import onnx_graphsurgeon as gs
import argparse
import onnx
import numpy as np

number_classes = 11 # change it 

def create_and_add_plugin_node(graph, topK, keepTopK):
    batch_size = graph.inputs[0].shape[0]
    print("The batch size is: ", batch_size)
    input_h = graph.inputs[0].shape[2]
    input_w = graph.inputs[0].shape[3]

    tensors = graph.tensors()
    boxes_tensor = tensors["boxes"]
    confs_tensor = tensors["scores"]

    # NMS Outputs
    num_detections = gs.Variable(
        name="num_detections", 
        dtype=np.int32, 
        shape=[batch_size, 1],
    )
    nmsed_boxes = gs.Variable(
        name="nmsed_boxes", 
        dtype=np.float32, 
        shape=[batch_size, keepTopK, 4],
    )
    nmsed_scores = gs.Variable(
        name="nmsed_scores",
        dtype=np.float32,
        shape=[batch_size, keepTopK],
    )
    nmsed_classes = gs.Variable(
        name="nmsed_classes", 
        dtype=np.float32, 
        shape=[batch_size, keepTopK],
    )

    new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

    nms_node = gs.Node(
        op="BatchedNMS_TRT",
        name="batched_nms",
        attrs=create_attrs(topK, keepTopK),
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs)

    graph.nodes.append(nms_node)
    graph.outputs = new_outputs

    return graph.cleanup().toposort()


def create_attrs(topK, keepTopK):
    global number_classes
    # num_anchors = 3

    # h1 = input_h // 8
    # h2 = input_h // 16
    # h3 = input_h // 32

    # w1 = input_w // 8
    # w2 = input_w // 16
    # w3 = input_w // 32

    # num_boxes = num_anchors * (h1 * w1 + h2 * w2 + h3 * w3)

    attrs = {}

    attrs["shareLocation"] = 1
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = number_classes
    attrs["topK"] = topK  # number of bounding boxes for nms eg 1000s
    attrs["keepTopK"] = keepTopK  # bounding boxes to be kept per image eg 20
    attrs["scoreThreshold"] = 0.25  # 0.70
    attrs["iouThreshold"] = 0.45
    attrs["isNormalized"] = 0
    attrs["clipBoxes"] = 0
    attrs['scoreBits'] = 16

    # 001 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs


def main():
    parser = argparse.ArgumentParser(description="Add batchedNMSPlugin")
    parser.add_argument("-f", "--model", help="Path to the ONNX model generated by export_model.py",
                        default="./postprocessed_model.onnx")
    parser.add_argument("-t", "--topK", help="number of bounding boxes for nms", default=100)
    parser.add_argument("-k", "--keepTopK", help="bounding boxes to be kept per image", default=100)

    args, _ = parser.parse_known_args()

    graph = gs.import_onnx(onnx.load(args.model))

    graph = create_and_add_plugin_node(graph, int(args.topK), int(args.keepTopK))

    onnx.save(gs.export_onnx(graph), args.model[:-5] + "_nms.onnx")
    print("batchedNMSPlugin was added successfully")


if __name__ == '__main__':
    main()
