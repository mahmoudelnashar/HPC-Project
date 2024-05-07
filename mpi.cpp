#include <iostream>
#include <chrono>
#include <string>
#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the image path from the command-line argument
    string imagePath;

    // Rank 0 reads the image path from the command-line argument
    if (world_rank == 0) {
        if (argc > 1) {
            imagePath = argv[1];
            cout << "Image path received by process " << world_size << ": " << imagePath << endl;
        }
        else {
            imagePath = "C:\\Users\\hody_\\OneDrive\\Documents\\HPCProject\\build\\Newfolder\\GrayscaleSegmentation.jpg";
        }
    }

    int k;
    int ImageWidth, ImageHeight;
    vector<int> red, green, blue;

    // Load image in rank 0 only
    Mat image_original;
    if (world_rank == 0) {
        // Load the image using OpenCV
        image_original = imread(imagePath, IMREAD_COLOR);
        if (image_original.empty()) {
            cerr << "Error: Unable to load image " << imagePath << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Get image dimensions
        ImageWidth = image_original.cols;
        ImageHeight = image_original.rows;

        // Resize vectors
        red.resize(ImageHeight * ImageWidth);
        green.resize(ImageHeight * ImageWidth);
        blue.resize(ImageHeight * ImageWidth);

        // Fill vectors
        for (int y = 0; y < ImageHeight; y++) {
            for (int x = 0; x < ImageWidth; x++) {
                Vec3b pixel = image_original.at<Vec3b>(y, x);
                red[y * ImageWidth + x] = pixel[2]; // OpenCV uses BGR ordering
                green[y * ImageWidth + x] = pixel[1];
                blue[y * ImageWidth + x] = pixel[0];
            }
        }
    }

    // Broadcast image dimensions from rank 0 to all other ranks
    MPI_Bcast(&ImageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ImageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "Image width by process " << world_rank << ": " << ImageWidth << endl;
        cout << "Image height by process " << world_rank << ": " << ImageHeight << endl;
    }
    // Broadcast kernel size from rank 0 to all other ranks
    if (world_rank == 0) {
        cout << "Enter the kernel size K:" << endl;
        cin >> k;
    }
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = chrono::high_resolution_clock::now();

    // Define the low-pass filter kernel
    vector<vector<int>> kernel(k, vector<int>(k, 1));

    int padding = k / 2;

    // Allocate memory for local image data
    vector<int> local_Red(ImageWidth * ImageHeight / world_size);
    vector<int> local_Green(ImageWidth * ImageHeight / world_size);
    vector<int> local_Blue(ImageWidth * ImageHeight / world_size);

    // Scatter image data to all processes
    MPI_Scatter(red.data(), ImageWidth * ImageHeight / world_size, MPI_INT, local_Red.data(), ImageWidth * ImageHeight / world_size,
        MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(green.data(), ImageWidth * ImageHeight / world_size, MPI_INT, local_Green.data(), ImageWidth * ImageHeight / world_size,
        MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(blue.data(), ImageWidth * ImageHeight / world_size, MPI_INT, local_Blue.data(), ImageWidth * ImageHeight / world_size,
        MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "Number of pixels per process: " << ImageWidth * ImageHeight / world_size << endl;
    }
    // Apply padding to local image data
    vector<vector<int>> local_padded_Red(ImageHeight / world_size + 2 * padding* ImageHeight/world_size, vector<int>(ImageWidth + 2 * padding * ImageWidth / world_size, 0));
    vector<vector<int>> local_padded_Green(ImageHeight / world_size + 2 * padding * ImageHeight/world_size, vector<int>(ImageWidth + 2 * padding * ImageWidth / world_size, 0));
    vector<vector<int>> local_padded_Blue(ImageHeight / world_size + 2 * padding * ImageHeight/world_size, vector<int>(ImageWidth + 2 * padding * ImageWidth / world_size, 0));

    if (world_rank == 0) {
        cout << "Number of padded pixels per process: " << ((ImageWidth+2*padding*ImageWidth)* (ImageWidth + 2 * padding * ImageWidth))/world_size<< endl;
    }

    // Copy data to local padded arrays, with edges copied for padding
    for (int i = 0; i < ImageHeight / world_size; i++) {
        for (int j = 0; j < ImageWidth; j++) {
            int index_padded_i = i + padding;
            int index_padded_j = j + padding;
            int index_local = i * ImageWidth + j;

            // Copy interior pixels
            local_padded_Red[index_padded_i][index_padded_j] = local_Red[index_local];
            local_padded_Green[index_padded_i][index_padded_j] = local_Green[index_local];
            local_padded_Blue[index_padded_i][index_padded_j] = local_Blue[index_local];

            // Copy edge pixels considering padding 
            if (i < 2*padding || i >= (ImageHeight / world_size) + 2*padding || j < 2*padding || j >= ImageWidth + 2*padding) {
                // Copy edge pixels
                int edge_i = (i < padding) ? padding : ((i >= (ImageHeight / world_size) + padding) ? (ImageHeight / world_size) + padding - 1 : index_padded_i);
                int edge_j = (j < padding) ? padding : ((j >= ImageWidth + padding) ? ImageWidth + padding - 1 : index_padded_j);

                local_padded_Red[index_padded_i][index_padded_j] = local_Red[edge_i * ImageWidth + edge_j];
                local_padded_Green[index_padded_i][index_padded_j] = local_Green[edge_i * ImageWidth + edge_j];
                local_padded_Blue[index_padded_i][index_padded_j] = local_Blue[edge_i * ImageWidth + edge_j];
            }
        }
    }

    cout << "Padding finished" << endl;

    // Apply the low-pass filter to local padded image data
    for (int y = padding; y < ImageHeight / world_size + padding; ++y) {
        for (int x = padding; x < ImageWidth + padding; ++x) {
            int redSum = 0, greenSum = 0, blueSum = 0;
            for (int ky = 0; ky < k; ++ky) {
                for (int kx = 0; kx < k; ++kx) {
                    redSum += kernel[ky][kx] * local_padded_Red[y + ky - padding][x + kx - padding];
                    greenSum += kernel[ky][kx] * local_padded_Green[y + ky - padding][x + kx - padding];
                    blueSum += kernel[ky][kx] * local_padded_Blue[y + ky - padding][x + kx - padding];
                }
            }
            // Divide by the number of elements in the kernel for averaging
            redSum /= (k * k);
            greenSum /= (k * k);
            blueSum /= (k * k);

            // Update local image data with filtered values
            int local_y = y - padding;
            int local_x = x - padding;
            local_Red[local_y * ImageWidth + local_x] = redSum;
            local_Green[local_y * ImageWidth + local_x] = greenSum;
            local_Blue[local_y * ImageWidth + local_x] = blueSum;
        }
    }
    cout << "Blurring finished" << endl;

    cout << "About to gather the image data" << endl;
    // Gather filtered image data from all processes
    MPI_Gather(local_Red.data(), ImageWidth * ImageHeight / world_size, MPI_INT, red.data(), ImageWidth * ImageHeight / world_size,
        MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_Green.data(), ImageWidth * ImageHeight / world_size, MPI_INT, green.data(), ImageWidth * ImageHeight / world_size,
        MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_Blue.data(), ImageWidth * ImageHeight / world_size, MPI_INT, blue.data(), ImageWidth * ImageHeight / world_size,
        MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Gathering finished" << endl;
    cout << "Almost done" << endl;

    // Rank 0 saves the filtered image
    if (world_rank == 0) {
        // Reconstruct the filtered image
        Mat filtered_image(ImageHeight, ImageWidth, CV_8UC3);
        for (int y = 0; y < ImageHeight; y++) {
            for (int x = 0; x < ImageWidth; x++) {
                filtered_image.at<Vec3b>(y, x) = Vec3b(blue[y * ImageWidth + x], green[y * ImageWidth + x],
                    red[y * ImageWidth + x]);
            }
        }


        // Calculate and output execution time
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "Execution time for Parallel Low Pass Filter using MPI is: " << duration.count() << " milliseconds"
            << endl;

        cout << "Finished" << endl;
        // Show the filtered image
        imshow("Filtered Image", filtered_image);
        waitKey(0); // Wait for a key press before closing the window
    }

    // Finalize MPI
    MPI_Finalize();


    return 0;
}
