// #include "yolov7.h"

using namespace sample;
using namespace std;
using namespace cv;

// Pretreatment
void preprocess(const cv::Mat& img, float data[]) {
	int w;
	int h;
	int x;
	int y;
	double r_w = INPUT_W / (img.cols*1.0);
	double r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar const* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
}

// just map the box back to the original image
std::vector<Bbox> rescale_box(const std::vector<Bbox> &out, int width, int height) {
	float gain = static_cast<float>(INPUT_SIZE) / std::max(width, height);
	float pad_x = (static_cast<float>(INPUT_W) - width * gain) / 2;
	float pad_y = (static_cast<float>(INPUT_W)  - height * gain) / 2;

	std::vector<Bbox> boxs;
	Bbox box;
	for (auto const & i : out) {
		box.x = (i.x - pad_x) / gain;
		box.y = (i.y - pad_y) / gain;
		box.w = i.w / gain;
		box.h = i.h / gain;
		box.score = i.score;
		box.classes = i.classes;

		boxs.push_back(box);
	}
	return boxs;
}

// visualization
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
	for (const auto &rect : bboxes)
	{
		cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
		cv::Scalar color = class_colors[rect.classes];
		cv::rectangle(image, rst, color, 2, cv::LINE_8, 0);

		int baseLine;
		std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";

		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
		rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - round(1.5*labelSize.height)),
			cv::Point(rect.x - rect.w / 2 + round(1.0*labelSize.width), rect.y - rect.h / 2 + baseLine), color, cv::FILLED);
		cv::putText(image, label, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	return image;
}

/*
This code is for working with videos. It displays the video during execution and
then saves it to your computer in the folder that you build the project
*/
void video_inference(std::string video_filepath, std::string output_name) {
	cv::VideoCapture video(video_filepath);
	if (!video.isOpened()) {
		std::cerr << "Failed to open video file." << std::endl;
		return 1;
	}

	auto frame_size = cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT));
	double fps = video.get(cv::CAP_PROP_FPS);
	cv::VideoWriter output_video(output_name, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, frame_size);

	cv::Mat frame;
	while (video.read(frame)){
		std::cout << "Processing frame..." << std::endl;

		cv::Mat image_origin = frame.clone();
		preprocess(frame, h_input);

		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
		cudaMalloc(&buffers[1], 1 * sizeof(int)); //<- num_detections
		cudaMalloc(&buffers[2], 1 * 100 * 4 * sizeof(float)); //<- nmsed_boxes
		cudaMalloc(&buffers[3], 1 * 100 * sizeof(float)); //<- nmsed_scores
		cudaMalloc(&buffers[4], 1 * 100 * sizeof(float)); //<- nmsed_classes

		cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // -- do execute --------//
        engine_context->executeV2(buffers);

        cudaMemcpy(h_output_0, buffers[1], 1 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_1, buffers[2], 1 * 100 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_2, buffers[3], 1 * 100 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_3, buffers[4], 1 * 100 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "THE COUNT OF DETECTION IN THIS FRAME: " << h_output_0[0] << std::endl;
        std::vector<Bbox> pred_box;

        for (int i = 0; i < h_output_0[0]; i++) {

            Bbox box;
            box.x = (h_output_1[i * 4 + 2] + h_output_1[i * 4]) / 2.0;
            box.y = (h_output_1[i * 4 + 3] + h_output_1[i * 4 + 1]) / 2.0;
            box.w = h_output_1[i * 4 + 2] - h_output_1[i * 4];
            box.h = h_output_1[i * 4 + 3] - h_output_1[i * 4 + 1];
            box.score = h_output_2[i];
            box.classes = (int)h_output_3[i];

            std::cout << "class: " << class_names[box.classes] << ", probability: " << box.score * 100 << "%" << std::endl;

            pred_box.push_back(box);
        }
        std::cout << std::endl;

        std::vector<Bbox> out = rescale_box(pred_box, frame.cols, frame.rows);
        cv::Mat img = renderBoundingBox(frame, out);
        cv::namedWindow("Video", 1);
        cv::imshow("Video", img);
        cv::waitKey(1);

        output_video.write(img);

        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(buffers[2]);
        cudaFree(buffers[3]);
        cudaFree(buffers[4]);
    }
    output_video.release();
}

/*
This code is for working with images. It displays images during code execution and then saves it to
your computer in the 'res' folder, which is created where you store image samples.
*/
void image_inference(cv::String pattern) {
    std::vector<cv::String> fn;
    cv::glob(pattern, fn, false);
    std::vector<cv::Mat> images;
    size_t count = fn.size(); //number of png files in images folder

    std::cout << count << std::endl;

    std::vector<cv::Mat> image_list; // Массив изображений для дальнейшнего сохраения
    std::vector<std::string> image_name_list;
    for (size_t i = 0; i < count; i++)
    {
        cv::Mat image = cv::imread(fn[i]);
        cv::Mat image_origin = image.clone();

        ////cv2读图片
        std::cout << fn[i] << std::endl; // ../samples_images/shot0001.png
        image_name_list.push_back(fn[i]);

        preprocess(image, h_input);

        cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
        cudaMalloc(&buffers[1], 1 * sizeof(int)); //<- num_detections
        cudaMalloc(&buffers[2], 1 * 100 * 4 * sizeof(float)); //<- nmsed_boxes
        cudaMalloc(&buffers[3], 1 * 100 * sizeof(float)); //<- nmsed_scores
        cudaMalloc(&buffers[4], 1 * 100 * sizeof(float)); //<- nmsed_classes

        cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // -- do execute --------//
        engine_context->executeV2(buffers);

        cudaMemcpy(h_output_0, buffers[1], 1 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_1, buffers[2], 1 * 100 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_2, buffers[3], 1 * 100 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_3, buffers[4], 1 * 100 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "THE COUNT OF DETECTION IN THIS FRAME: " << h_output_0[0] << std::endl;
        std::vector<Bbox> pred_box;
        for (int i = 0; i < h_output_0[0]; i++) {

            Bbox box;
            box.x = (h_output_1[i * 4 + 2] + h_output_1[i * 4]) / 2.0;
            box.y = (h_output_1[i * 4 + 3] + h_output_1[i * 4 + 1]) / 2.0;
            box.w = h_output_1[i * 4 + 2] - h_output_1[i * 4];
            box.h = h_output_1[i * 4 + 3] - h_output_1[i * 4 + 1];
            box.score = h_output_2[i];
            box.classes = (int)h_output_3[i];

            std::cout << "class: " <<class_names[box.classes] << ", probability: " << box.score * 100 << "%" << std::endl;

            pred_box.push_back(box);
        }

        std::vector<Bbox> out = rescale_box(pred_box, image.cols, image.rows);
        cv::Mat img = renderBoundingBox(image, out);
        cv::namedWindow("Image", 1);
        cv::imshow("Image", img);
        cv::waitKey(0); // waiting for a response from the user

        image_list.push_back(img);

        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(buffers[2]);
        cudaFree(buffers[3]);
        cudaFree(buffers[4]);
    }

    for (int i = 0; i < image_list.size(); ++i) {
        int pos = image_name_list[i].find_last_of('.');

        std::string rst_name = image_name_list[i].insert(pos, "_inference");
        rst_name = rst_name.insert(image_name_list[i].find_last_of('/') + 1, "res/");

        std::cout << "image " <<  rst_name << " was saved" << std::endl;
        cv::imwrite(rst_name, image_list[i]);
    }
}

int main()
{
	Logger gLogger;
	// to initialize the plugin, you must initialize the plugin respo when calling the plugin
	initLibNvInferPlugins(&gLogger, "");

	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	std::string engine_filepath = "../tensorrt_cpp_inference/1_model_NMS.trt"; // or *.engine

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	std::streamoff length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
	nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

	int input_index = engine_infer->getBindingIndex("images"); //1x3x640x640
	//std::string input_name = engine_infer->getBindingName(0)
	int output_index_1 = engine_infer->getBindingIndex("num_detections");  //1
	int output_index_2 = engine_infer->getBindingIndex("nmsed_boxes");   // 2
	int output_index_3 = engine_infer->getBindingIndex("nmsed_scores");  //3
	int output_index_4 = engine_infer->getBindingIndex("nmsed_classes"); //5

	std::cout << "images: " << input_index << " num_detections-> " << output_index_1 << " nmsed_boxes-> " << output_index_2
		<< " nmsed_scores-> " << output_index_3 << " nmsed_classes-> " << output_index_4 << std::endl;

	if (engine_context == nullptr)
	{
		std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
	}

	std::cout << "loaded trt model , do inference" << std::endl;

	if (mode == "video") {
        video_inference("../samples_video/video.avi", "output.mp4");
	} else if (mode == "image") {
		image_inference("../samples_images/*.png");
	} else {
		std::cout << "Please, choose the mode" << std::endl;
		return 1;
	}

	engine_runtime->destroy();
	engine_infer->destroy();
	return 0;
}
