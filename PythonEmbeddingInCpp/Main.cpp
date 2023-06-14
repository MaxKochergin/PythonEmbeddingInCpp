#include <pybind11/embed.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

int main() {
    py::scoped_interpreter python;
    //Тест 1 - запуск функции извлечения корня
    auto math = py::module::import("math");
    double root_two = math.attr("sqrt")(2.0).cast<double>();

    std::cout << "The square root of 2 is: " << root_two << "\n";

    //Тест 2 - запуск дерева решений из Sklearn

    //импортируем numpy
    auto numpy = py::module::import("numpy");

    //наши данные
    // x = [1, 2, 3, 4]
    // y = [5, 4, 1, 0]
    int rows = 4;
    double* x1 = new double[rows]{ 1, 2, 3, 4 };
    double* x2 = new double[rows] { 5, 4, 1, 0 };
    int* labes = new int[rows] {0, 0, 1, 1};

    //sklearn.tree.DecisionTreeClassifier() принимает данные вида [[x1, y1],[x2,y2],...]
    // где x, y - разные признаки
    //создаём два списка, чтобы записать их в numpy-матрицу
    //список 1
    auto arrayList1 = py::list();
    for (int i = 0; i < rows; i++)
        arrayList1.append(x1[i]);
    //numpy-массив 1
    auto arrayNumpy_1 = numpy.attr("array")(arrayList1);
    //список 2
    auto arrayList2 = py::list();
    for (int i = 0; i < rows; i++)
        arrayList2.append(x2[i]);
    //numpy-массив 2
    auto arrayNumpy_2 = numpy.attr("array")(arrayList2);
    //массив меток
    auto arrayLabels = py::list();
    for (int i = 0; i < rows; i++)
        arrayLabels.append(labes[i]);
    //формируем из [x1,x2,...] и [y1,y2,...] numpy-массив вида [[x1, y1],[x2,y2],...]
    py::tuple args = py::make_tuple(arrayNumpy_1, arrayNumpy_2);
    auto arrayNumpy = numpy.attr("vstack")(args);//конкатенация
    arrayNumpy = arrayNumpy.attr("T");//транспонирование

    py::print("We get input data: ");
    py::print(arrayNumpy);

    //импортируем собственную функцию из PythonWrapper.py, которая вызывает
    //функцию fit (обучение) из tree.DecisionTreeClassifier
    py::function TreeFit =
        py::reinterpret_borrow<py::function>(
            py::module::import("PythonWrapper").attr("TreeFit")
        );
    //получаем из python обученную модель
    auto model = TreeFit(arrayNumpy, arrayLabels);
    //задаём данные для тестирования
    double* testData = new double[2]{3, 1};
    auto arrayTest = py::list();
    for (int i = 0; i < 2; i++)
        arrayTest.append(testData[i]);
    py::print("We gonna test model on one point:");
    py::print(arrayTest);
    //наша точка имеет вид [x1, y1]
    //а predict принимает [[x1, y1]]
    auto listTest = py::list();
    listTest.insert(0,arrayTest);
    //выводим то, что подаём
    py::print("Sending data to Python: ");
    py::print(listTest);
    auto result = model.attr("predict")(listTest);
    std::string message = "Point " + std::string(py::str(listTest)) + " belongs to class: ";
    py::print(message);
    py::print(result);


}