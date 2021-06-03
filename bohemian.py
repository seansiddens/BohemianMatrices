import numpy as np
import random 
import math
import cmath
from numpy.linalg import LinAlgError
from tqdm import tqdm
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt


# Converts a point on the complex plane to an integer 
# (x, y) coordinate within an image w/ a width and height
def complex_to_image(z, width, height, real_range, imag_range, centered_at):
    bottom_left = centered_at - complex(real_range / 2, imag_range / 2)
    # Find x coordinate
    x = (z.real - bottom_left.real) / real_range # Normalized
    x *= width

    # Find y coordinate
    y = (z.imag - bottom_left.imag) / imag_range # Normalized
    y = 1 - y
    y *= height

    return (int(x), int(y))

# Plot of eigenvalues of random 10x10 tridiagonal matrices
def tridiagonal(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 10 # n x n matrix
    mono_coloring = False

    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    # Non-zero elements of the matrix are sampled from this sequence
    values = np.array([complex(0, 0), complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1), 
                    complex(20, 0), complex(-20, 0), complex(0, 20), complex(0, -20)])

    # Complute eigenvalues of samples number of matrices 
    print("Computing eigenvalues...")
    for _ in tqdm(range(samples)):
        # Entries not in the sub-, super-, or main-diagonals are zero
        mat = np.full((n, n), complex(0, 0)) 
        # Set entries in sub- and super-diagonal
        for i in range(1, n): 
            mat[i, i-1] = random.choice(values)
            mat[i-1, i] = random.choice(values)
        # Set entries along main diagonal
        for i in range(n):
            mat[i, i] = random.choice(values)
        
        # Get eigvenvalues of the matrix
        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            print("Eigenvalue error")


        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            # Check if it's in image
            if (0 <= x < width) and (0 <= y < height):
                counts[y, x] += 1
    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x] != 0:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    brightness = math.log(counts[y, x]) / math.log(max_count)
                    gamma = 2.2
                    brightness = math.pow(brightness, 1/gamma)
                    rgba = cmap(brightness)
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    return im


# Returns image of "eigenfish", plot of eigenvalues of matrices w/ entries in
# {-1, 0, 1}, and some entries taking on random continous values in a specified range
def eigenfish(im_arr, width, height, real_range, imag_range, centered_at):
    # A and B are at (2, 2) and (3, 3) respectively, and take on values
    # in (3, 3)
    mat = np.array([[0, 0, -1, -1],
                    [1, -1, 0, 0], 
                    [0, 1, 0, 0],
                    [1, 0, -1, 0]], dtype=float)

    for n in tqdm(range(50000)):
        # Set A and B to random number in (-3, 3)
        A = random.uniform(-3, 3)
        B = random.uniform(-3, 3)
        mat[2, 2] = A
        mat[3, 3] = B

        # Get eigvenvalues of the matrix
        eigvals = np.linalg.eigvals(mat)

        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            if z.imag == 0:
                continue
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            # Check if it's in image
            if (0 <= x < width) and (0 <= y < height):
                im_arr[y, x] = 255

    im = Image.fromarray(im_arr)
    return im


# Plot of eigenvalues of companion matrices of littlewood polynomials. 
# Coefficients take on values in {-1, 1}
def littlewood(im_arr, width, height, real_range, imag_range, centered_at, cmap, input_file=None, degree=14):
    mono_coloring = False
    print("Rendering Littlewood fractal...")
    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    # Read roots from input file instead of computing them
    if input_file is not None:
        print("Reading roots from input file...")
        with open(input_file) as file:
            for line in tqdm(file):
                line = line.split(" ")
                z = complex(float(line[0]), float(line[1]))
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                if (0 <= x < width) and (0 <= y < height):
                    counts[y, x] += 1
    else: 
        coefficients = np.zeros((degree), dtype=np.int16) # Coefficients are all initialized to -1

        # Initialize the companion matrix
        companion = np.zeros((degree-1, degree-1), dtype=float) 
        for i in range(1, degree-1):
            companion[i, i-1] = 1 # Set sub-diagonal to 1's
        
        # Iterate through every permutation of coefficients and
        # compute the eigenvalues of the companion matrix    
        print("Computing eigenvalues...")
        for n in tqdm(range(pow(2, degree))):
            for i in range(degree):
                # Essentially we are counting from binary, so we retrieve each bit of the
                # current iteration number and map it to -1 or 1
                coefficients[i] = -1 if (1 & (n >> i) == 1) else 1

            if coefficients[degree-1] == 1:
                monic = True
            else:
                monic = False

            # Construct companion matrix from polynomial
            # Final column is (-a_0, -a_1, ..., -a_(n-1))
            # If polynomial is not monic, invert every coefficient to make it monic
            for i in range(degree-1):
                a = coefficients[i]
                a = -a if not monic else a
                companion[i, -1] = -a

            # Get eigenvalues of the copmpanion matrix
            eigvals = np.linalg.eigvals(companion)

            # Convert each eigvenvalue to a point on image
            for z in eigvals:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                # Check if it's in image
                if (0 <= x < width) and (0 <= y < height):
                    if z.imag != 0:
                        counts[y, x] += 1

    # Get maximum eigenvalue count in the array
    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x] != 0:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else:
                    brightness = math.log(counts[y, x]) / math.log(max_count)
                    gamma = 2.2
                    brightness = math.pow(brightness, 1/gamma)
                    rgba = cmap(brightness)
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    return im


# Computes all complex roots of all Littlewood polynomials of specified degree,
# and writes them out to a file
def compute_littlewood_roots(degree, outfile):
    fd = open(outfile, "w")
    coefficients = np.zeros((degree), dtype=np.int16) # Coefficients are all initialized to -1

    # Initialize the companion matrix
    companion = np.zeros((degree-1, degree-1), dtype=float) 
    for i in range(1, degree-1):
        companion[i, i-1] = 1 # Set sub-diagonal to 1's
    
    # Iterate through every permutation of coefficients and
    # compute the eigenvalues of the companion matrix    
    print("Computing eigenvalues...")
    for n in tqdm(range(pow(2, degree))):
        for i in range(degree):
            # Essentially we are counting from binary, so we retrieve each bit of the
            # current iteration number and map it to -1 or 1
            coefficients[i] = -1 if (1 & (n >> i) == 1) else 1

        if coefficients[degree-1] == 1:
            monic = True
        else:
            monic = False

        # Construct companion matrix from polynomial
        # Final column is (-a_0, -a_1, ..., -a_(n-1))
        # If polynomial is not monic, invert every coefficient to make it monic
        for i in range(degree-1):
            a = coefficients[i]
            a = -a if not monic else a
            companion[i, -1] = -a

        # Get eigenvalues of the copmpanion matrix
        eigvals = np.linalg.eigvals(companion)

        # Write out each complex eigenvalue to the outfile in the form:
        # real imag
        for z in eigvals:
            if z.imag == 0:
                continue
            fd.write(str(z.real) + " " + str(z.imag) + "\n") 

    fd.close()


if __name__ == "__main__":
    # Size of complex window w/ respect to a center focal point
    centered_at = complex(0, 0)
    real_offset = (-1, 1)
    imag_offset = (-1, 1)
    real_range = real_offset[1] - real_offset[0]
    imag_range = imag_offset[1] - imag_offset[0]
    scale = 48
    real_range *= scale
    imag_range *= scale

    # Set width and height of image depending on aspect ratio
    aspect_ratio = real_range / imag_range
    width = 1024 * 4
    height = int(width * 1 / aspect_ratio)

    # # Initialize image array
    im_arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Number of samples we want to take from set of matrices
    samples = 10000000

    # # Initialize color map
    # cmap = cm.get_cmap("jet")
    # cmap = cm.get_cmap("viridis")
    # cmap = cm.get_cmap("nipy_spectral")
    cmap = cm.get_cmap("hot")
    
    # im = eigenfish(im_arr, width, height, real_range, imag_range, centered_at)
    # im = littlewood(im_arr, width, height, real_range, imag_range, centered_at, cmap)
    im = tridiagonal(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples)

    print("Saving image...")
    im.save("out.png")
    print("Done!")
