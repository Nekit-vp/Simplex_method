import numpy as np


def continue_solve(mark_in):  # проверка положительных оценок
    mark = np.copy(mark_in)
    mark = mark[1:]
    for i in mark:
        if i > 0:
            return True,
    return False


def get_mark(matrix, function, basis):  # вычисление оценки
    c_basis = []
    for i in basis:
        c_basis.append(function[i - 1])
    mark = np.dot(c_basis, matrix) - (np.append([0], function))
    print(mark)
    print('-----------------')
    return mark


def get_basis(matrix):  # получение базиса
    basis = []
    for i in range(len(matrix)):
        basis.append(matrix.shape[1] - len(matrix) + i)
    return basis


def add_additional_variables(matrix, function):  # добавление переменных к матрице и функции
    matrix = np.concatenate((matrix, np.eye(matrix.shape[0])), axis=1)
    function = np.append(function, matrix.shape[0] * [0])
    return matrix, function


def recount(matrix_in, index_input, index_output):  # пересчет мартрицы
    matrix = matrix_in.copy()
    k = matrix[index_output][index_input]
    matrix[index_output] /= k

    for i in range(len(matrix)):
        if i != index_output:
            matrix[i] -= matrix[i][index_input] * matrix[index_output]
    print(matrix)
    print('-----------------')
    return matrix


def get_index_input(mark):
    return np.argmax(mark)


def get_index_output(index_input, matrix_in):
    matrix = np.copy(matrix_in)
    p_0 = matrix[:, 0]
    p_i = matrix[:, index_input]

    p_i[p_i == 0] = -1  # exclude division by zero

    teta = p_0 / p_i
    teta = np.where(teta > 0, teta, np.inf)
    index_output = teta.argmin()

    if teta[index_output] == np.inf:
        raise Exception("Not solution")
    else:
        return index_output


def solve(matrix, function, basis):
    print(matrix)
    print("-----------------")
    mark = get_mark(matrix, function, basis)
    flag = continue_solve(mark)

    while flag:  # main loop

        index_input = get_index_input(mark)
        index_output = get_index_output(index_input, matrix)

        matrix = recount(matrix, index_input, index_output)

        basis[index_output] = index_input

        mark = get_mark(matrix, function, basis)
        flag = continue_solve(mark)

    return matrix, function, basis


def canonization(a, b, c):
    matrix = np.copy(a)
    vector = np.copy(b)
    function = np.copy(c * -1)

    matrix = np.concatenate((vector.T, matrix), axis=1)
    matrix, function = add_additional_variables(matrix, function)
    basis = get_basis(matrix)

    return matrix, function, basis


def simplex_method(matrix, function, basis):
    matrix, function, basis = solve(matrix, function, basis)
    mark = get_mark(matrix, function, basis)

    p_0 = matrix[:, 0]

    x = np.zeros(len(C))

    for i in range(len(basis)):
        if (basis[i] - 1) < len(C):
            x[basis[i] - 1] = p_0[i]

    print("x = " + str(x))
    print("result = " + str(mark[0] * -1))


A = np.array([[6, 4, 2],
              [2, 3, 6]], dtype=np.float)
B = np.array([[68, 60]], dtype=np.float)
C = np.array([10, 15, 15])

# A = np.array([[1, -2],
#               [1, -1]], dtype=np.float)
# B = np.array([[1, 1]], dtype=np.float)
# C = np.array([1, 4])


# A = np.array([[-1, 1],
#               [0, 1],
#               [1, 0]], dtype=np.float)
# B = np.array([[2, 1, 3]], dtype=np.float)
# C = np.array([6, 10])

mat, fun, bas = canonization(A, B, C)
simplex_method(mat, fun, bas)
