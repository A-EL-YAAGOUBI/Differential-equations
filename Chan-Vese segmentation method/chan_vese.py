# Chan-Vese segmentation method implementation
# Authors :
# Anass El Yaagoubi
# Victor Le Maistre
# Simon Delecourt


def square(x):
    """Squares the number x.

    Parameters
    ----------
    x : float
        Number to be squared.

    Returns
    -------
    float
        Square of x.
    """

    return x*x


def nablaPlus(x_n, i, j, axis=0):
    """Computes the forward derivative.

    Parameters
    ----------
    x_n : np.array
        Surface to be differentiated.
    i : int
        Y position.
    j : int
        X position.
    axis : int
        Dimension to be differentiated (default axis=0).

    Returns
    -------
    float
        Derivative of x_n at position (i, j).

    """

    if axis == 0:
        if j == x_n.shape[1]-1:
            return 0
        else:
            return (x_n[i, j+1] - x_n[i, j])/2

    if axis == 1:
        if i == x_n.shape[0]-1:
            return 0
        else:
            return (x_n[i+1, j] - x_n[i, j])/2


def nablaMinus(x_n, i, j, axis=0):
    """Computes the backward derivative.

    Parameters
    ----------
    x_n : np.array
        Surface to be differentiated.
    i : int
        Y position.
    j : int
        X position.
    axis : int
        Dimension to be differentiated (default axis=0).

    Returns
    -------
    float
        Derivative of x_n at position (i, j).

    """

    if axis == 0:
        if j == 0:
            return 0
        else:
            return (x_n[i, j-1] - x_n[i, j])/2

    if axis == 1:
        if i == 0:
            return 0
        else:
            return (x_n[i-1, j] - x_n[i, j])/2


def getPhi(phi, i, j):
    """Gets value of phi at position (i, j), and avoids getting outside of the domaine.

    Parameters
    ----------
    phi : np.array
        phi matrix.
    i : int
        Position of Y.
    j : int
        Position of X.

    Returns
    -------
    float
        Value of phi at position (i, j).
    """

    return phi[min(max(i, 0), phi.shape[0] - 1), min(max(j, 0), phi.shape[1] - 1)]


def nablaZero(x_n, i, j, axis=0):
    """Computes centered derivative at position (i, j).

    Parameters
    ----------
    x_n : np.array
        Surface to be differentiated.
    i : int
        Y position.
    j : int
        X position.
    axis : int
        Dimension to be differentiated (default axis=0).

    Returns
    -------
    float
        Centered derivative of x_n at position (i, j).
    """

    return (nablaPlus(x_n, i, j, axis) + nablaMinus(x_n, i, j, axis)) / 2


def A(phi, i, j, mu=0.2, eta=1e-8):
    """Added notation to simplify computations and to structure the code
    see : http://www.ipol.im/pub/art/2012/g-cv/article.pdf.

    Parameters
    ----------
    phi : np.array
        2D function (from R² to R).
    i : int
        Y position.
    j : int
        X position.
    mu : float
        Length penalization parameter.
    eta : float
        Regularization parameter for the curvature (avoids division by zero).

    Returns
    -------
    float
        value of A at position (i, j).
    """

    import numpy as np
    return mu / np.sqrt(square(eta) + square(nablaPlus(phi, i, j, axis=1)) + square(nablaZero(phi, i, j, axis=0)))


def B(phi, i, j, mu=0.2, eta=1e-8):
    """Added notation to simplify computations and to structure the code
    see : http://www.ipol.im/pub/art/2012/g-cv/article.pdf.

    Parameters
    ----------
    phi : np.array
        2D function (from R² to R).
    i : int
        Y position.
    j : int
        X position.
    mu : float
        Length penalization parameter.
    eta : float
        Regularization parameter for the curvature (avoids division by zero).

    Returns
    -------
    float
        value of B at position (i, j).
    """

    import numpy as np
    return mu / np.sqrt(square(eta) + square(nablaPlus(phi, i, j, axis=0)) + square(nablaZero(phi, i, j, axis=1)))


def deltaRegularized(x, epsilon=1):
    """Derivative of the Heaviside function, i-e Dirac Mass.

    Parameters
    ----------
    x : float
        Position x.
    epsilon : float
        Regularizer of the Dirac Mass.

    Returns
    -------
    float
        Value of the regularized Dirac Mass.
    """

    import math
    return epsilon / (math.pi*(square(epsilon) + square(x)))


def initPhi(x):
    """initialization of phi.

    Parameters
    ----------
    x : np.array
        Matrix of the size of phi.

    Returns
    -------
    np.array
        Initial Phi as Checkboard function.
    """

    import math
    import numpy as np
    res = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i, j] = math.sin(math.pi/5 * i)*math.sin(math.pi/5 * j)
    return res


def u(img, phi):
    """Computes U from Phi.

    Parameters
    ----------
    img : np.array
        image.
    phi : np.array
        function phi.

    Returns
    -------
    np.array
        Piecewise constant funciton U.
    """

    import numpy as np
    c1 = np.mean(img[phi > 0])
    c2 = np.mean(img[phi <= 0])

    res = np.zeros(img.shape)
    res[phi > 0] = c1
    res[phi <= 0] = c2
    return c1, c2, res


def updatePhi(f, c1, c2, phi, dt=0.5, nu=0, lambda1=1, lambda2=1):
    """Update step that realizes the chanvese method.

    Parameters
    ----------
    f : np.array
        Image tp be segmented.
    c1 : float
        average value of phi inside the contour (i-e first value of U).
    c2 : float
        average value of phi outside the contour (i-e second value of U).
    phi : np.array
        Function Phi to be updated.
    dt : float
        Time step subdivision.
    nu : float
        Length penalization parameter.
    lambda1 : float
        Penalization parameter for approximation error inside the contour.
    lambda2 : float
        Penalization parameter for approximation error outside the contour.

    Returns
    -------
    np.array
        New Phi.
    """

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            numerator = getPhi(phi, i, j) + dt * deltaRegularized(getPhi(phi, i, j)) * (A(phi, i, j) * getPhi(phi, i - 1, j) + B(phi, i, j) * getPhi(phi, i, j + 1) + B(phi, i, j - 1) * getPhi(phi, i, j - 1) - nu - lambda1 * square(f[i, j] - c1) + lambda2 * square(f[i, j] - c2))
            denominator = 1 + dt*deltaRegularized(getPhi(phi, i, j))*(A(phi, i, j) + A(phi, i - 1, j) + B(phi, i, j) + B(phi, i, j - 1))
            phi[i, j] = numerator / denominator
    return phi


def computeChanVase(f, ITER_MAX=20, tol=1e-1):
    """Chan-Vese segmentation method.

    Parameters
    ----------
    f : np.array
        Image tp be segmented.
    ITER_MAX : int
        Maximum number of iterations.
    tol : float
        Stoping parameter.

    Returns
    -------
    list(np.array)
        Successive values of U accross iterations.
    """

    import numpy as np
    print('Chan-Vese method :')
    images = []
    phi      = initPhi(f)
    last_phi = phi.copy() + 1
    phis = [phi]
    i = 0
    while i < ITER_MAX and np.linalg.norm(last_phi - phi) > tol:
        i += 1
        if i % 5 == 0:
            print('Iteration {}'.format(i))
        c1, c2, img = u(f, phi)
        images.append(img)
        last_phi = np.copy(phi)
        phi = updatePhi(f, c1, c2, phi)
        phis.append(phi)

    # return i, images, phis
    return images
