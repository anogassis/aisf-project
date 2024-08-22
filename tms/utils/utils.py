import pickle
import glob
import numpy as np

def generate_2d_kgon_vertices(k, rot=0., pad_to=None, force_length=0.9):
    """
    Generate the vertices of a 2D k-gon.

    Parameters
    ----------
    k : int
        The number of sides of the k-gon.
    rot : float, optional
        The rotation angle in radians, by default 0.
    pad_to : int, optional
        If specified, pads the vertices with zeros to match the specified number of sides, by default None.
    force_length : float, optional
        The scaling factor applied to the vertices, by default 0.9.

    Returns
    -------
    numpy.ndarray
        An array of shape (2, k) containing the x and y coordinates of the vertices.
    """
    # Angles for the vertices
    theta = np.linspace(0, 2*np.pi, k, endpoint=False) + rot

    # Generate the vertices
    x = np.cos(theta)
    y = np.sin(theta)
    result = np.vstack((x, y))

    if pad_to is not None and k < pad_to:
        num_pad = pad_to - k
        result = np.hstack([result, np.zeros((2, num_pad))])

    return result * force_length

def generate_init_param(m, n, init_kgon, prior_std=1., no_bias=True, init_zerobias=True, seed=0, force_negb=False, noise=0.01):
    """
    Generate initial parameters for a neural network layer.

    Parameters
    ----------
    m : int
        Number of output units.
    n : int
        Number of input units.
    init_kgon : int or None
        Number of vertices for the 2D k-gon initialization. If None or m != 2, a normal initialization is used.
    prior_std : float, optional
        Standard deviation of the normal distribution used for initialization (default is 1.0).
    no_bias : bool, optional
        Whether to include bias in the initialization (default is True).
    init_zerobias : bool, optional
        Whether to initialize the bias with zeros (default is True).
    seed : int, optional
        Seed for the random number generator (default is 0).
    force_negb : bool, optional
        Whether to force the bias to be negative (default is False).
    noise : float, optional
        Standard deviation of the noise added to the initialization (default is 0.01).

    Returns
    -------
    dict
        A dictionary containing the initialized parameters:
        - "W" : numpy.ndarray
            The weight matrix of shape (m, n).
        - "b" : numpy.ndarray, optional
            The bias vector of shape (n, 1), only included if `no_bias` is False.
    """
    np.random.seed(seed)

    if init_kgon is None or m != 2:
        init_W = np.random.normal(size=(m, n)) * prior_std
    else:
        assert init_kgon <= n
        rand_angle = np.random.uniform(0, 2 * np.pi, size=(1,))
        noise = np.random.normal(size=(m, n)) * noise
        init_W = generate_2d_kgon_vertices(init_kgon, rot=rand_angle, pad_to=n) + noise

    if no_bias:
        param = {"W": init_W}
    else:
        init_b = np.random.normal(size=(n, 1)) * prior_std
        if force_negb:
            init_b = -np.abs(init_b)
        if init_zerobias:
            init_b = init_b * 0
        param = {
            "W": init_W,
            "b": init_b
        }
    return param

def generate_optimal_solution(m, n, rot=0.0):
    """
    Generate the optimal solution parameters for a given m and n.

    Parameters
    ----------
    m : int
        The value of m.
    n : int
        The value of n.
    rot : float, optional
        The rotation parameter, by default 0.0.

    Returns
    -------
    dict
        A dictionary containing the optimal solution parameters.

    Raises
    ------
    AssertionError
        If m is not equal to 2.
        If n is not equal to 6.

    Notes
    -----
    Solutions exist for multiples of 4 and 5, 6, and 7.

    If n is equal to 6, the optimal length parameter is 1.4142, but the paper suggests 1.32053.
    If n is equal to 6, the optimal bias parameter is -0.9999 instead of 0.61814.

    """
    assert m == 2
    assert n == 6

    if n == 6:
        l = 1.4142
        init_b = -np.ones((n, 1)) * 0.9999

    init_w = generate_2d_kgon_vertices(n, rot=rot, force_length=l, pad_to=n)
    param = {
        "W": init_w,
        "b": init_b
    }
    return param

def generate_sparsity_values(scale: float, count: int) -> np.ndarray:
    """
    Generate sparsity values using an exponential decay function.

    Parameters
    ----------
    scale : float
        The scale parameter for the exponential decay function.
    count : int
        The number of values to generate.

    Returns
    -------
    numpy.ndarray
        An array of sparsity values generated using the exponential decay function.
    """
    # Generate exponential values from 0 to scale
    x = np.linspace(0, scale, count)
    # Apply the exponential decay function
    values = 1 - np.exp(-x)
    return values

def load_results(data_dir, version="1.5.0"):
    """
    Load results from pickle files.

    Parameters
    ----------
    data_dir : str
        The directory where the pickle files are located.
    version : str, optional
        The version of the results to load, by default "1.5.0".

    Returns
    -------
    list
        A list of loaded results.

    Raises
    ------
    FileNotFoundError
        If no files matching the file pattern are found.
    """
    file_pattern = f'{data_dir}/logs_loss_{version}_*.pkl'
    results = []

    for file_path in glob.glob(file_pattern):
        try:
            with open(file_path, "rb") as file:
                results.append(pickle.load(file))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            print(type(e))
            break

    if not results:
        raise FileNotFoundError(f"No files matching the pattern {file_pattern} found.")

    return results
