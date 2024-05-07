#include <QApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QErrorMessage>
#include <QFileDialog>
#include <QFormLayout>
#include <QImage>
#include <QInputDialog>
#include <QLabel>
#include <QLayout>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QPixmap>
#include <QSpinBox>
#include <QString>
#include <QVBoxLayout>
#include <QDebug>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include "mainwindow.h"
#include "ui_mainwindow.h"



using namespace std;
namespace fs = std::filesystem;

MainWindow::MainWindow(QWidget *parent)

    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    // Initialize MPI
    MPI_Init(NULL, NULL);
    // Get the number of processes and the rank of the current process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    ui->setupUi(this);
    this->setWindowTitle("Team 4 HPC Project");
    setGeometry(100, 100, 640, 480);

    // Layout
    QVBoxLayout *layout = new QVBoxLayout(this);

    // Set layout to central widget
    QWidget *centralWidget = new QWidget(this);
    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);

    // Image Label
    imageLabel = new QLabel(this);
    imageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // Allow the label to expand
    imageLabel->setAlignment(Qt::AlignCenter); // Align the label's content to center
    layout->addWidget(imageLabel);

    // Button to browse image
    QPushButton *browseImageButton = new QPushButton("Browse Image", this);
    layout->addWidget(browseImageButton);
    connect(browseImageButton, &QPushButton::clicked, this, &MainWindow::onBrowseImageClicked);


    // Button to save image
    QPushButton *saveImageButton= new QPushButton("Save Image",this);
    layout->addWidget(saveImageButton);
    connect(saveImageButton, &QPushButton::clicked,this,&MainWindow::onSaveImageClicked);

    // Button to reset filter
    QPushButton *ResetFilterButton = new QPushButton("Reset Applied Filter", this);
    layout->addWidget(ResetFilterButton);
    connect(ResetFilterButton, &QPushButton::clicked, this, &MainWindow::onResetImageClicked);


    // Button for low pass filter
    QPushButton *lowPassFilterButton = new QPushButton("Apply Low-Pass Filter (Blurring)", this);
    layout->addWidget(lowPassFilterButton);
    connect(lowPassFilterButton, &QPushButton::clicked, this, &MainWindow::applyLowPassFilter);


    // Button for high pass filter
    QPushButton *highPassFilterButton = new QPushButton("Apply High-Pass Filter (Sharpening)", this);
    layout->addWidget(highPassFilterButton);
    connect(highPassFilterButton, &QPushButton::clicked, this, &MainWindow::applyHighPassFilter);



    // Label to display time
    outputLabel = new QLabel(this);
    outputLabel->setVisible(false);
    layout->addWidget(outputLabel);

    // Shared variables
    red = nullptr;
    blue = nullptr;
    green = nullptr;
    input_name="";
    imagePath="";

    maximum_threrads=omp_get_max_threads();

}

MainWindow::~MainWindow()
{

    MPI_Finalize();
    // Free memory allocated for the arrays
    delete[] red;
    delete[] blue;
    delete[] green;
    delete ui;

}



//****************************************Image Handling**************************************************************

//                                        Load Image
void MainWindow::onBrowseImageClicked()
{
    imagePath = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.jpeg *.png *.jpg *.bmp)"));
    if (!imagePath.isEmpty()) {
        // Load image
        qDebug()<<imagePath;
        QImage image_original(imagePath);
        if (!image_original.isNull()) {
            QPixmap pixmap_original = QPixmap::fromImage(image_original);
            imageLabel->setPixmap(pixmap_original);
            ImageWidth = pixmap_original.width();
            ImageHeight = pixmap_original.height();

            // Allocate memory for colored arrays
            red = new int[ImageHeight * ImageWidth];
            blue = new int[ImageHeight * ImageWidth];
            green = new int[ImageHeight * ImageWidth];

            // Fill arrays
            for (int y = 0; y < ImageHeight; y++) {
                for (int x = 0; x < ImageWidth; x++) {
                    QRgb pixel = image_original.pixel(x, y);
                    red[y * ImageWidth + x] = qRed(pixel);
                    blue[y * ImageWidth + x] = qBlue(pixel);
                    green[y * ImageWidth + x] = qGreen(pixel);
                }
            }
        }
        outputLabel->setVisible(true);
    }
}





//                                        Save Image
void MainWindow::onSaveImageClicked()
{
    if (red != nullptr && green != nullptr && blue != nullptr) {
        // Open file dialog to select the save location
        QString filePath = QFileDialog::getSaveFileName(this, tr("Save Image"), input_name+"_output", tr("Images (*.png *.jpg *.bmp)"));

        if (!filePath.isEmpty()) {
            // Retrieve the pixmap from the label
            QPixmap pixmap = imageLabel->pixmap().copy();

            // Save the pixmap to the selected file path
            if (pixmap.save(filePath)) {
                // Update the label with the output
                outputLabel->setText("Image saved to: " + filePath);
            } else {
                qDebug() << "Failed to save image.";
                QMessageBox::critical(this, "Error", "Failed to save image.");
            }
        }


    } else {
        qDebug() << "No image loaded.";
        QMessageBox messageBox;
        messageBox.critical(0,"Error","No Image Loaded");
        messageBox.setFixedSize(500,200);    }
}



//                                        Reset Image
void MainWindow::onResetImageClicked()
{
    if (red != nullptr && green != nullptr && blue != nullptr) {
        // Create an empty QImage to hold the combined RGB image
        QImage combinedImage(ImageWidth, ImageHeight, QImage::Format_RGB888);

        // Copy pixel values from each channel array to the combined QImage
        for (int y = 0; y < ImageHeight; ++y) {
            for (int x = 0; x < ImageWidth; ++x) {
                int index = y * ImageWidth + x;
                // Set RGB pixel value in the combined image
                QRgb pixelValue = qRgb(red[index], green[index], blue[index]);
                combinedImage.setPixel(x, y, pixelValue);
            }
        }

        // Convert the QImage to a QPixmap
        QPixmap pixmap = QPixmap::fromImage(combinedImage);

        // Display the QPixmap in the label or do any further processing as needed
        imageLabel->setPixmap(pixmap);
        //imageLabel->setScaledContents(true);
        //imageLabel->adjustSize();

        // Update the label with the output
        outputLabel->setText("Filter Resetted");


    } else {
        qDebug() << "No image loaded.";
        QMessageBox messageBox;
        messageBox.critical(0,"Error","No Image Loaded");
        messageBox.setFixedSize(500,200);    }
}


























//******************************************Low Pass Filter***********************************************************

//                                          Choose Computation
void MainWindow::applyLowPassFilter()
{
    if (red != nullptr && green != nullptr && blue != nullptr) {
        QMessageBox messageBox;
        messageBox.setWindowTitle("Computation Type");
        messageBox.setText("Choose Computation Type For Blurring");

        messageBox.setStandardButtons(QMessageBox::Cancel);
        messageBox.setEscapeButton(QMessageBox::Cancel);

        QAbstractButton *sequential =
            messageBox.addButton(tr("Sequential"), QMessageBox::ActionRole);
        QAbstractButton *openmp =
            messageBox.addButton(tr("Openmp"), QMessageBox::ActionRole);
        QAbstractButton *mpi =
            messageBox.addButton(tr("MPI"), QMessageBox::ActionRole);
        messageBox.exec();

        if (messageBox.clickedButton() == sequential) {
            sequentiallowpassfilter();
        }
        else if (messageBox.clickedButton() == openmp) {
            openmplowpassfilter();
        }
        else if  (messageBox.clickedButton() == mpi) {
            mpilowpassfilter();
        }
        else{
            qDebug() << "No choice made";
        }


    } else {
        qDebug() << "No image loaded.";
        QMessageBox messageBox;
        messageBox.critical(0,"Error","No Image Loaded");
        messageBox.setFixedSize(500,200);    }
}


//                                          Sequential Low Pass
void MainWindow::sequentiallowpassfilter(){
    qDebug()<<"sequential";
    int k = QInputDialog::getInt(this, tr("Enter Kernel Size"), QString("Kernel Size (KxK):"), 3, 3, 19, 2);
    auto start = chrono::high_resolution_clock::now();

    // Define the low-pass filter kernel
    // Create a kxk filter array filled with ones
    int **kernel = new int*[k];
    for (int i = 0; i < k; ++i) {
        kernel[i] = new int[k];
        for (int j = 0; j < k; ++j) {
            kernel[i][j] = 1;
        }
    }
    // Calculate padding size
    int padding = k / 2;

    // Apply the low-pass filter to each color channel separately
    QImage filteredImage(ImageWidth, ImageHeight, QImage::Format_RGB32);

    // Apply zero padding to the image
    int *paddedRed = new int[(ImageWidth + 2 * padding) * (ImageHeight + 2 * padding)];
    int *paddedGreen = new int[(ImageWidth + 2 * padding) * (ImageHeight + 2 * padding)];
    int *paddedBlue = new int[(ImageWidth + 2 * padding) * (ImageHeight + 2 * padding)];

    // Copy the original image to the padded image
    for (int y = padding; y < ImageHeight + padding; ++y) {
        for (int x = padding; x < ImageWidth + padding; ++x) {
            paddedRed[y * (ImageWidth + 2 * padding) + x] = red[(y - padding) * ImageWidth + (x - padding)];
            paddedGreen[y * (ImageWidth + 2 * padding) + x] = green[(y - padding) * ImageWidth + (x - padding)];
            paddedBlue[y * (ImageWidth + 2 * padding) + x] = blue[(y - padding) * ImageWidth + (x - padding)];
        }
    }

    // Apply the low-pass filter with zero padding
    for (int y = padding; y < ImageHeight + padding; ++y) {
        for (int x = padding; x < ImageWidth + padding; ++x) {
            int redSum = 0, greenSum = 0, blueSum = 0;
            for (int ky = 0; ky < k; ++ky) {
                for (int kx = 0; kx < k; ++kx) {
                    redSum += kernel[ky][kx] * paddedRed[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                    greenSum += kernel[ky][kx] * paddedGreen[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                    blueSum += kernel[ky][kx] * paddedBlue[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                }
            }
            // Divide by the number of elements in the kernel for averaging
            redSum /= (k * k);
            greenSum /= (k * k);
            blueSum /= (k * k);

            QRgb color = qRgb(redSum, greenSum, blueSum);
            filteredImage.setPixel(x - padding, y - padding, color);
        }
    }

    // Convert the QImage to a QPixmap
    QPixmap pixmap = QPixmap::fromImage(filteredImage);

    // Display the QPixmap in the label or do any further processing as needed
    imageLabel->setPixmap(pixmap);
    //imageLabel->setScaledContents(true);
    //imageLabel->adjustSize();

    // end timer
    auto end = chrono::high_resolution_clock::now();
    // calculate duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(end -start);

    QString durationString = QString::number(duration.count());
    // Update the label with the output
    outputLabel->setText("Execution time for Sequential Low Pass Filter on a single thread is: " + durationString + " milliseconds");

    // Free memory
    for (int i = 0; i < k; ++i) {
        delete[] kernel[i];
    }
    delete[] kernel;
    delete[] paddedRed;
    delete[] paddedGreen;
    delete[] paddedBlue;
}



//                                          Openmp Low Pass
void MainWindow::openmplowpassfilter(){
    qDebug() << "openmp";

    QDialog dialog(this);
    dialog.setWindowTitle("OpenMP Low Pass Filter Configuration");

    QFormLayout formLayout(&dialog);

    QSpinBox kernelSizeSpinBox;
    kernelSizeSpinBox.setRange(3, 99);
    kernelSizeSpinBox.setValue(3);
    kernelSizeSpinBox.setSingleStep(2);
    QLabel kernelSizeLabel("Kernel Size (KxK):");
    formLayout.addRow(&kernelSizeLabel, &kernelSizeSpinBox);

    int maximum_threads = omp_get_max_threads();
    QSpinBox numThreadsSpinBox;
    numThreadsSpinBox.setRange(2, maximum_threads);
    numThreadsSpinBox.setValue(2);
    QLabel numThreadsLabel("Number of Threads:");
    formLayout.addRow(&numThreadsLabel, &numThreadsSpinBox);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QObject::connect(&buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(&buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    formLayout.addRow(&buttonBox);

    if (dialog.exec() == QDialog::Accepted) {
        int k = kernelSizeSpinBox.value();
        int num_of_threads = numThreadsSpinBox.value();
        auto start = chrono::high_resolution_clock::now();

        // Define the low-pass filter kernel
        int **kernel = new int*[k];
        for (int i = 0; i < k; ++i) {
            kernel[i] = new int[k];
            for (int j = 0; j < k; ++j) {
                kernel[i][j] = 1;
            }
        }

        // Calculate padding size
        int padding = k / 2;

        // Apply the low-pass filter to each color channel separately
        QImage filteredImage(ImageWidth, ImageHeight, QImage::Format_RGB32);

        // Apply zero padding to the image
        int *paddedRed = new int[(ImageWidth + 2 * padding) * (ImageHeight + 2 * padding)];
        int *paddedGreen = new int[(ImageWidth + 2 * padding) * (ImageHeight + 2 * padding)];
        int *paddedBlue = new int[(ImageWidth + 2 * padding) * (ImageHeight + 2 * padding)];

// Copy the original image to the padded image
#pragma omp parallel for num_threads(num_of_threads)
        for (int y = padding; y < ImageHeight + padding; ++y) {
            for (int x = padding; x < ImageWidth + padding; ++x) {
                paddedRed[y * (ImageWidth + 2 * padding) + x] = red[(y - padding) * ImageWidth + (x - padding)];
                paddedGreen[y * (ImageWidth + 2 * padding) + x] = green[(y - padding) * ImageWidth + (x - padding)];
                paddedBlue[y * (ImageWidth + 2 * padding) + x] = blue[(y - padding) * ImageWidth + (x - padding)];
            }
        }

// Apply the low-pass filter with zero padding in parallel
#pragma omp parallel for num_threads(num_of_threads)
        for (int y = padding; y < ImageHeight + padding; ++y) {
            for (int x = padding; x < ImageWidth + padding; ++x) {
                int redSum = 0, greenSum = 0, blueSum = 0;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        redSum += kernel[ky][kx] * paddedRed[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                        greenSum += kernel[ky][kx] * paddedGreen[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                        blueSum += kernel[ky][kx] * paddedBlue[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                    }
                }
                redSum /= (k * k); // Divide by the number of elements in the kernel for averaging
                greenSum /= (k * k);
                blueSum /= (k * k);

                QRgb color = qRgb(redSum, greenSum, blueSum);
                filteredImage.setPixel(x - padding, y - padding, color);
            }
        }

        // Convert the QImage to a QPixmap
        QPixmap pixmap = QPixmap::fromImage(filteredImage);

        // Display the QPixmap in the label or do any further processing as needed
        imageLabel->setPixmap(pixmap);

        // end timer
        auto end = chrono::high_resolution_clock::now();
        // calculate duration
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        QString durationString = QString::number(duration.count());
        QString threadsString = QString::number(num_of_threads);

        // Update the label with the output
        outputLabel->setText("Openmp execution time for Parallel Low Pass Filter using {"+threadsString+"} threads is: " + durationString + " milliseconds");

        // Free memory
        for (int i = 0; i < k; ++i) {
            delete[] kernel[i];
        }
        delete[] kernel;
        delete[] paddedRed;
        delete[] paddedGreen;
        delete[] paddedBlue;
    }
}


//                                          mpi Low Pass
void MainWindow::mpilowpassfilter() {
    // Get current directory
    fs::path currentPath = fs::current_path();

    int n = QInputDialog::getInt(this, tr("Enter Number of Processes for MPI"), QString("MPI Processes: "), 1, 1, 12, 1);

    // Construct batch file path
    string batchFilePath = (currentPath / "run_mpi_low_pass_filter.bat").string();

    // Create and open the batch file
    ofstream batchFile(batchFilePath);
    if (!batchFile.is_open()) {
        qCritical() << "Failed to create batch file!";
        return;
    }

    // Convert the QString to std::string
    string imagePathStr = imagePath.toStdString();


    // Write commands to the batch file
    batchFile << "@echo off" << endl;
    batchFile << "mpiexec -n "<< n<< " \"mpi_lowpassfilter.exe\" " << "\""<<imagePathStr <<"\""<< endl;

    // Close the batch file
    batchFile.close();

    // Execute the batch file
    string command = "cmd /C \"" + batchFilePath + "\"";
    system(command.c_str());
}
























//******************************************High Pass Filter***********************************************************

//                                          Choose Computation
void MainWindow::applyHighPassFilter()
{
    if (red != nullptr && green != nullptr && blue != nullptr) {
        QMessageBox messageBox;
        messageBox.setWindowTitle("Computation Type");
        messageBox.setText("Choose Computation Type For Sharpenning");

        messageBox.setStandardButtons(QMessageBox::Cancel);
        messageBox.setEscapeButton(QMessageBox::Cancel);

        QAbstractButton *sequential =
            messageBox.addButton(tr("Sequential"), QMessageBox::ActionRole);
        QAbstractButton *openmp =
            messageBox.addButton(tr("Openmp"), QMessageBox::ActionRole);
        QAbstractButton *mpi =
            messageBox.addButton(tr("MPI"), QMessageBox::ActionRole);
        messageBox.exec();

        if (messageBox.clickedButton() == sequential) {
            sequentialhighpassfilter();
        }
        else if (messageBox.clickedButton() == openmp) {
            openmphighpassfilter();
        }
        else if (messageBox.clickedButton() == mpi) {
            mpihighpassfilter();
        }
        else{
            qDebug() << "No choice made";
        }


    } else {
        qDebug() << "No image loaded.";
        QMessageBox messageBox;
        messageBox.critical(0,"Error","No Image Loaded");
        messageBox.setFixedSize(500,200);    }
}


//                                          Sequential High Pass
void MainWindow::sequentialhighpassfilter(){
    qDebug()<<"sequential";
    int k = QInputDialog::getInt(this, tr("Enter Kernel Size"), QString("Kernel Size (KxK):"), 3, 3, 19, 2);
    auto start = chrono::high_resolution_clock::now();

    // Define the high-pass filter kernel
    vector<vector<int>> kernel(k, vector<int>(k, -1)); // Create a kxk filter array filled with negative 1
    kernel[k/2][k/2]=k*k-1;


    // Calculate padding size
    int padding = k / 2;

    // Apply the high-pass filter to each color channel separately
    QImage filteredImage(ImageWidth, ImageHeight, QImage::Format_RGB32);

    // Apply zero padding to the image
    QVector<int> paddedRed((ImageWidth + 2 * padding) * (ImageHeight + 2 * padding));
    QVector<int> paddedGreen((ImageWidth + 2 * padding) * (ImageHeight + 2 * padding));
    QVector<int> paddedBlue((ImageWidth + 2 * padding) * (ImageHeight + 2 * padding));

    // Copy the original image to the padded image
    for (int y = padding; y < ImageHeight + padding; ++y) {
        for (int x = padding; x < ImageWidth + padding; ++x) {
            paddedRed[y * (ImageWidth + 2 * padding) + x] = red[(y - padding) * ImageWidth + (x - padding)];
            paddedGreen[y * (ImageWidth + 2 * padding) + x] = green[(y - padding) * ImageWidth + (x - padding)];
            paddedBlue[y * (ImageWidth + 2 * padding) + x] = blue[(y - padding) * ImageWidth + (x - padding)];
        }
    }

    // Apply the high-pass filter with zero padding
    for (int y = padding; y < ImageHeight + padding; ++y) {
        for (int x = padding; x < ImageWidth + padding; ++x) {
            int redSum = 0, greenSum = 0, blueSum = 0;
            for (int ky = 0; ky < k; ++ky) {
                for (int kx = 0; kx < k; ++kx) {
                    redSum += kernel[ky][kx] * paddedRed[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                    greenSum += kernel[ky][kx] * paddedGreen[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                    blueSum += kernel[ky][kx] * paddedBlue[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                }
            }

            redSum = qBound(0, redSum, 255); // Ensure the pixel value is within [0, 255]
            greenSum = qBound(0, greenSum, 255);
            blueSum = qBound(0, blueSum, 255);
            QRgb color = qRgb(redSum, greenSum, blueSum);
            filteredImage.setPixel(x - padding, y - padding, color);
        }
    }

    // Convert the QImage to a QPixmap
    QPixmap pixmap = QPixmap::fromImage(filteredImage);

    // Display the QPixmap in the label or do any further processing as needed
    imageLabel->setPixmap(pixmap);
    //imageLabel->setScaledContents(true);
    //imageLabel->adjustSize();

    // end timer
    auto end = chrono::high_resolution_clock::now();
    // calculate duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(end -start);

    QString durationString = QString::number(duration.count());
    // Update the label with the output
    outputLabel->setText("Execution time for Sequential High Pass Filter on a single thread is: " + durationString + " milliseconds");

}


//                                          openmp High Pass
void MainWindow::openmphighpassfilter(){
    qDebug() << "openmp";

    QDialog dialog(this);
    dialog.setWindowTitle("OpenMP Filter Configuration");

    QFormLayout formLayout(&dialog);

    QSpinBox kernelSizeSpinBox;
    kernelSizeSpinBox.setRange(3, 99);
    kernelSizeSpinBox.setValue(3);
    kernelSizeSpinBox.setSingleStep(2);
    QLabel kernelSizeLabel("Kernel Size (KxK):");
    formLayout.addRow(&kernelSizeLabel, &kernelSizeSpinBox);

    int maximum_threads = omp_get_max_threads();
    QSpinBox numThreadsSpinBox;
    numThreadsSpinBox.setRange(2, maximum_threads);
    numThreadsSpinBox.setValue(2);
    QLabel numThreadsLabel("Number of Threads:");
    formLayout.addRow(&numThreadsLabel, &numThreadsSpinBox);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QObject::connect(&buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(&buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    formLayout.addRow(&buttonBox);

    if (dialog.exec() == QDialog::Accepted) {
        int k = kernelSizeSpinBox.value();
        int num_of_threads = numThreadsSpinBox.value();
        auto start = chrono::high_resolution_clock::now();

        // Define the high-pass filter kernel
        vector<vector<int>> kernel(k, vector<int>(k, -1)); // Create a kxk filter array filled with negative 1
        kernel[k/2][k/2]=k*k-1;

        // Calculate padding size
        int padding = k / 2;

        // Apply the high-pass filter to each color channel separately
        QImage filteredImage(ImageWidth, ImageHeight, QImage::Format_RGB32);

        // Apply zero padding to the image
        QVector<int> paddedRed((ImageWidth + 2 * padding) * (ImageHeight + 2 * padding));
        QVector<int> paddedGreen((ImageWidth + 2 * padding) * (ImageHeight + 2 * padding));
        QVector<int> paddedBlue((ImageWidth + 2 * padding) * (ImageHeight + 2 * padding));

// Copy the original image to the padded image
#pragma omp parallel for num_threads(num_of_threads)
        for (int y = padding; y < ImageHeight + padding; ++y) {
            for (int x = padding; x < ImageWidth + padding; ++x) {
                paddedRed[y * (ImageWidth + 2 * padding) + x] = red[(y - padding) * ImageWidth + (x - padding)];
                paddedGreen[y * (ImageWidth + 2 * padding) + x] = green[(y - padding) * ImageWidth + (x - padding)];
                paddedBlue[y * (ImageWidth + 2 * padding) + x] = blue[(y - padding) * ImageWidth + (x - padding)];
            }
        }

// Apply the high-pass filter with zero padding in parallel
#pragma omp parallel for num_threads(num_of_threads)
        for (int y = padding; y < ImageHeight + padding; ++y) {
            for (int x = padding; x < ImageWidth + padding; ++x) {
                int redSum = 0, greenSum = 0, blueSum = 0;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        redSum += kernel[ky][kx] * paddedRed[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                        greenSum += kernel[ky][kx] * paddedGreen[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                        blueSum += kernel[ky][kx] * paddedBlue[(y + ky - padding) * (ImageWidth + 2 * padding) + (x + kx - padding)];
                    }
                }

                redSum = qBound(0, redSum, 255); // Ensure the pixel value is within [0, 255]
                greenSum = qBound(0, greenSum, 255);
                blueSum = qBound(0, blueSum, 255);
                QRgb color = qRgb(redSum, greenSum, blueSum);
                filteredImage.setPixel(x - padding, y - padding, color);
            }
        }

        // Convert the QImage to a QPixmap
        QPixmap pixmap = QPixmap::fromImage(filteredImage);

        // Display the QPixmap in the label or do any further processing as needed
        imageLabel->setPixmap(pixmap);
        //imageLabel->setScaledContents(true);
        //imageLabel->adjustSize();

        // end timer
        auto end = chrono::high_resolution_clock::now();
        // calculate duration
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        QString durationString = QString::number(duration.count());
        QString threadsString = QString::number(num_of_threads);

        // Update the label with the output
        outputLabel->setText("Openmp execution time for Parallel High Pass Filter using {"+threadsString+"} threads is: " + durationString + " milliseconds");
    }
}


//                                          mpi High Pass
void MainWindow::mpihighpassfilter()
{
    // Get current directory
    fs::path currentPath = fs::current_path();

    int n = QInputDialog::getInt(this, tr("Enter Number of Processes for MPI"), QString("MPI Processes: "), 1, 1, 12, 1);

    // Construct batch file path
    string batchFilePath = (currentPath / "run_mpi_high_pass_filter.bat").string();

    // Create and open the batch file
    ofstream batchFile(batchFilePath);
    if (!batchFile.is_open()) {
        qCritical() << "Failed to create batch file!";
        return;
    }

    // Convert the QString to std::string
    string imagePathStr = imagePath.toStdString();


    // Write commands to the batch file
    batchFile << "@echo off" << endl;
    batchFile << "mpiexec -n "<< n<< " \"mpi_highpassfilter.exe\" " << "\""<<imagePathStr <<"\""<< endl;

    // Close the batch file
    batchFile.close();

    // Execute the batch file
    string command = "cmd /C \"" + batchFilePath + "\"";
    system(command.c_str());

}
















