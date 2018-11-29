"""
Tumor detection using Active Contour :

- Active contour is a method for image segmentation that uses edge detection. A contour is defined in the begining and then it is evolved in order to divide the image into segmments. Here is an implementation of the Chan-Vese method where we start with a checker board initialization (sinus function) and iterate over it in order to find the best segmentation. In this example we cannot apply the method to the whole image otherwise we will only get a segmentation of the brain and the background, so to remediate to this problem we only consider a small snippet that contains the tumor and do the segmentation on it. We thus obtain a shrinkage of 8.85% of the tumor.

- Les méthodes de Contour actif sont des méthodes de segmentation d’image qui se base sur la détection de contours. Un contour est défini au début, puis est évolué pour diviser l image en segments. Dans ce TP j utilise mon implémentation de la méthode Chan-Vese dans laquelle nous commençons par une initialisation du damier (fonction sinus) et itérons dessus pour trouver la meilleure segmentation. Dans cet exemple, nous ne pouvons pas appliquer la méthode à l’ensemble de l’image. Par conséquent, pour remédier à ce problème, considérons unse petite imagette contenant la tumeur et effectuons la segmentation correspondante. Nous obtenons ainsi un rétrécissement de 8,85% de la tumeur.

- https://github.com/A-EL-YAAGOUBI/Differential-equations/blob/master/Chan-Vese%20segmentation%20method/Chan_Vese_Segmentation_Presentation.pdf
"""




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


def nabla_plus(x_n, i, j, axis=0):
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
            return (x_n[i,j+1] - x_n[i,j])/2

    if axis == 1:
        if i == x_n.shape[0]-1:
            return 0
        else:
            return (x_n[i+1,j] - x_n[i,j])/2


def nabla_minus(x_n, i, j, axis=0):
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
            return (x_n[i,j-1] - x_n[i,j])/2

    if axis == 1:
        if i == 0:
            return 0
        else:
            return (x_n[i-1,j] - x_n[i,j])/2


def get_phi(phi, i, j):
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

    return phi[min(max(i,0),phi.shape[0]-1),min(max(j,0),phi.shape[1]-1)]


def nabla_zero(x_n, i, j,axis=0):
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

    return (nabla_plus(x_n, i, j, axis) + nabla_minus(x_n, i, j, axis)) / 2

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

    return mu / np.sqrt(square(eta) + square(nabla_plus(phi, i, j, axis=1)) + square(nabla_zero(phi, i, j, axis=0)))


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

    return mu / np.sqrt(square(eta) + square(nabla_plus(phi, i, j, axis=0)) + square(nabla_zero(phi, i, j, axis=1)))


def delta_regularized(x,epsilon=1):
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

    return epsilon / (math.pi*(square(epsilon) + square(x)))


def init_phi(x):
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

    res = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i,j] = math.sin(math.pi/5 * i)*math.sin(math.pi/5 * j)
    return res


def u(img,phi):
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

    c1 = np.mean(img[phi > 0])
    c2 = np.mean(img[phi <= 0])

    res = np.zeros(img.shape)
    res[phi > 0] = c1
    res[phi <= 0] = c2
    return c1,c2, res


def update_phi(f, c1, c2, phi, dt=0.5, nu = 0, lambda1 = 1 ,lambda2 = 1):
    """Update step that realizes the chan_vese method.

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
            numerator = get_phi(phi,i,j) + dt * delta_regularized(get_phi(phi,i,j)) * (A(phi,i,j) * get_phi(phi,i-1,j) + B(phi,i,j) * get_phi(phi,i,j+1) + B(phi,i,j-1) * get_phi(phi,i,j-1) - nu - lambda1 * square(f[i,j] - c1) + lambda2 * square(f[i,j] - c2))
            denominator = 1 + dt*delta_regularized(get_phi(phi,i,j))*(A(phi,i,j) + A(phi,i-1,j) + B(phi,i,j) + B(phi,i,j-1))
            phi[i,j] = numerator / denominator
    return phi


def chan_vese(f, ITER_MAX=20, tol=1e-1):
    print('Chan-Vese method :')
    images = []
    phi      = init_phi(f)
    last_phi = phi.copy() + 1

    phis = [phi]
    i = 0
    while i < ITER_MAX and np.linalg.norm(last_phi - phi) > tol:
        i += 1
        if i % 5 == 0:
            print('Iteration {}'.format(i))
        c1, c2, img = u(f,phi)
        images.append(img)
        last_phi = np.copy(phi)
        phi = update_phi(f, c1, c2, phi)
        phis.append(phi)
    return images #return i,images,phis
