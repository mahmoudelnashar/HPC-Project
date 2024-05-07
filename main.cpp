#include "mainwindow.h"
#include <mpi.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication::addLibraryPath("./");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
