#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void onBrowseImageClicked();
    void onSaveImageClicked();
    void onResetImageClicked();

    void applyLowPassFilter();
    void sequentiallowpassfilter();
    void openmplowpassfilter();
    void mpilowpassfilter();

    void applyHighPassFilter();
    void sequentialhighpassfilter();
    void openmphighpassfilter();
    void mpihighpassfilter();

    //void applyKmeansSegmentation();


private:
    Ui::MainWindow *ui;
    QLabel *imageLabel;
    QLabel *outputLabel;
    QTabWidget *tabWidget;
    QString input_name;
    QString imagePath;

    int maximum_threrads;
    int ImageWidth;
    int ImageHeight;
    int *red;
    int *blue;
    int *green;
};

#endif // MAINWINDOW_H
