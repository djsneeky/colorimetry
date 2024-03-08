import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CIE 1931 standard RGB primaries
cie_rgb_primitives = {
    'R': [0.73467, 0.26533],
    'G': [0.27376, 0.71741],
    'B': [0.16658, 0.00886]
}

# Rec. 709 RGB primaries
rec709_rgb_primitives = {
    'R': [0.640, 0.330],
    'G': [0.300, 0.600],
    'B': [0.150, 0.060]
}

# D65 white point
d65_white_point = [0.3127, 0.3290, 0.3583]

# Equal energy white point
equal_energy_white_point = [0.3333, 0.3333]

def main():
    data = np.load('data.npy', allow_pickle=True).item()
    x0 = data['x'][0]
    y0 = data['y'][0]
    z0 = data['z'][0]
    illum1 = data['illum1'][0]
    illum2 = data['illum2'][0]
    
    plt.style.use('Solarize_Light2')
    
    # plot_color_matching_functions(x0, y0, z0, illum1, illum2)
    # plot_chromaticity(x0, y0, z0)
    
    reflect = np.load('reflect.npy', allow_pickle=True).item()
    R_coeff = reflect['R']
    
    # image = render(R_coeff, illum1, x0, y0, z0)
    # image.save('images/illum1_image.tif')
    # image.save('images/illum1_image.png')
    # image = render(R_coeff, illum2, x0, y0, z0)
    # image.save('images/illum2_image.tif')
    # image.save('images/illum2_image.png')
    
    plot_monitor_chromaticity(x0, y0, z0)
    
    return 0

def plot_color_matching_functions(x0, y0, z0, illum1, illum2):
    wavelengths = np.arange(400, 701, 10)
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, x0, label='x0(λ)')
    plt.plot(wavelengths, y0, label='y0(λ)')
    plt.plot(wavelengths, z0, label='z0(λ)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Color Matching Functions')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/color_matching_vs_wavelength.png')
    
    # compute using transformation
    A_inv = np.array([[0.2430, 0.8560, -0.0440],
                    [-0.3910, 1.1650, 0.0870],
                    [0.0100, -0.0080, 0.5630]])
    l0, m0, s0 = np.dot(A_inv, np.array([x0, y0, z0]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, l0, label='l0(λ)')
    plt.plot(wavelengths, m0, label='m0(λ)')
    plt.plot(wavelengths, s0, label='s0(λ)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Cone Responses (l0(λ), m0(λ), s0(λ))')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/cone_vs_wavelength.png')

    # Plot the spectrum of the D65 and fluorescent illuminants versus wavelength
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, illum1, label='D65 Illuminant')
    plt.plot(wavelengths, illum2, label='Fluorescent Illuminant')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Illuminants Spectra')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/illum_vs_wavelength.png')
    
def plot_chromaticity(x0, y0, z0):
    # transform XYZ to (x,y)
    x = x0 / (x0 + y0 + z0)
    y = y0 / (x0 + y0 + z0)
    # z = z0 / (x0 + y0 + z0)

    plt.figure(figsize=(10, 10))
    plt.plot(x, y, label='Pure Spectral Source', color='black')

    # Plot CIE 1931 standard RGB primaries
    for key, value in cie_rgb_primitives.items():
        plt.scatter(value[0], value[1], color='red')
        plt.text(value[0], value[1], key)

    # Plot lines between the CIE 1931 standard RGB primaries
    plt.plot([cie_rgb_primitives['R'][0], cie_rgb_primitives['G'][0]], [cie_rgb_primitives['R'][1], cie_rgb_primitives['G'][1]], color='red', linestyle='--', label='CIE 1931')
    plt.plot([cie_rgb_primitives['G'][0], cie_rgb_primitives['B'][0]], [cie_rgb_primitives['G'][1], cie_rgb_primitives['B'][1]], color='red', linestyle='--')
    plt.plot([cie_rgb_primitives['B'][0], cie_rgb_primitives['R'][0]], [cie_rgb_primitives['B'][1], cie_rgb_primitives['R'][1]], color='red', linestyle='--')

    # Plot Rec. 709 RGB primaries
    for key, value in rec709_rgb_primitives.items():
        plt.scatter(value[0], value[1], color='blue')
        plt.text(value[0], value[1], key)

    # Plot lines between the CIE 1931 standard RGB primaries
    plt.plot([rec709_rgb_primitives['R'][0], rec709_rgb_primitives['G'][0]], [rec709_rgb_primitives['R'][1], rec709_rgb_primitives['G'][1]], color='blue', linestyle='--', label='Rec 709')
    plt.plot([rec709_rgb_primitives['G'][0], rec709_rgb_primitives['B'][0]], [rec709_rgb_primitives['G'][1], rec709_rgb_primitives['B'][1]], color='blue', linestyle='--')
    plt.plot([rec709_rgb_primitives['B'][0], rec709_rgb_primitives['R'][0]], [rec709_rgb_primitives['B'][1], rec709_rgb_primitives['R'][1]], color='blue', linestyle='--')

    # Plot D65 white point
    plt.scatter(d65_white_point[0], d65_white_point[1], color='green')
    plt.text(d65_white_point[0], d65_white_point[1], 'D65')

    # Plot equal energy white point
    plt.scatter(equal_energy_white_point[0], equal_energy_white_point[1], color='purple')
    plt.text(equal_energy_white_point[0], equal_energy_white_point[1], 'Equal Energy')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Chromaticity Diagram')
    plt.savefig('images/chromaticities.png')
    plt.show()
    
def render(R_coeff, illum, x0, y0, z0):
    # Using the D65 light source (illum1) and R, compute the reflected light energy at each given wavelength, λ, using equation (1)
    # Do this for each pixel in the image, producing an m × n × 31 array, called I.
    I_D65 = np.zeros_like(R_coeff)
    for i in range(R_coeff.shape[0]):
        for j in range(R_coeff.shape[1]):
            I_D65[i, j] = R_coeff[i, j] * illum
            
    XYZ = np.zeros((R_coeff.shape[0], R_coeff.shape[1], 3))
    for i in range(R_coeff.shape[0]):
        for j in range(R_coeff.shape[1]):
            XYZ[i, j, 0] = np.sum(x0 * I_D65[i, j])
            XYZ[i, j, 1] = np.sum(y0 * I_D65[i, j])
            XYZ[i, j, 2] = np.sum(z0 * I_D65[i, j])
            
    # Calculate the XYZ tristimulus values of the white point
    Xwp, Ywp, Zwp = d65_white_point

    # Calculate the XYZ tristimulus values for each primary using the scaling constants
    M709_D65 = np.zeros((3, 3))

    # Compute the scaling constants
    xr, yr = rec709_rgb_primitives['R']
    xg, yg = rec709_rgb_primitives['G']
    xb, yb = rec709_rgb_primitives['B']

    # Construct the transformation matrix and its inverse
    M_rgb_to_xyz = np.array([
        [xr, xg, xb],
        [yr, yg, yb],
        [1 - xr - yr, 1 - xg - yg, 1 - xb - yb]
    ])
    M_xyz_to_rgb = np.linalg.inv(M_rgb_to_xyz)

    # Compute the scaling constants
    kr, kg, kb = np.dot(M_xyz_to_rgb, [Xwp/Ywp, 1, Zwp/Ywp])

    # Construct the transformation matrix
    M709_D65[0, :] = [kr * xr, kg * xg, kb * xb]
    M709_D65[1, :] = [kr * yr, kg * yg, kb * yb]
    M709_D65[2, :] = [kr * (1 - xr - yr), kg * (1 - xg - yg), kb * (1 - xb - yb)]

    # Print the transformation matrix
    print("Transformation matrix M_709_D65:")
    print(M709_D65)
    
    # Transform XYZ to RGB
    RGB = np.zeros_like(XYZ)
    for i in range(XYZ.shape[0]):
        for j in range(XYZ.shape[1]):
            RGB[i, j] = np.dot(np.linalg.inv(M709_D65), XYZ[i, j])

    # Clip RGB component values outside [0, 1]
    RGB = np.clip(RGB, 0, 1)
    
    # Gamma correction
    gamma = 2.2
    RGB_corrected = RGB ** (1 / gamma)
    
    # Scale pixel values and convert to uint8
    RGB_scaled = (RGB_corrected * 255).astype(np.uint8)
    
    # Display the result using Pillow
    image = Image.fromarray(RGB_scaled)
    
    return image

def plot_monitor_chromaticity(x0, y0, z0):
    # Step 1: Create chromaticity matrices
    x, y = np.meshgrid(np.arange(0, 1.005, 0.005), np.arange(0, 1.005, 0.005))
    z = 1 - x - y

    # Step 2: Compute RGB values for each chromaticity coordinate
    # Use Rec. 709 RGB primaries
    xr, yr = rec709_rgb_primitives['R']
    xg, yg = rec709_rgb_primitives['G']
    xb, yb = rec709_rgb_primitives['B']

    # Construct the transformation matrix and its inverse
    M709 = np.array([
        [xr, xg, xb],
        [yr, yg, yb],
        [1 - xr - yr, 1 - xg - yg, 1 - xb - yb]
    ])
    M709_inv = np.linalg.inv(M709)

    # Reshape the meshgrid coordinates to prepare for matrix multiplication
    xyz_coords = np.stack((x.flatten(), y.flatten(), z.flatten()))

    # Transform xyz to RGB using matrix multiplication
    RGB_flat = np.dot(M709_inv, xyz_coords)

    # Reshape the RGB values back to the shape of the meshgrid
    rgb_image = RGB_flat.reshape(3, x.shape[0], x.shape[1])
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

    # Clip RGB values
    rgb_image[rgb_image < 0] = 0
    rgb_image[rgb_image > 1] = 1

    # Gamma correction
    gamma = 2.2
    rgb_corrected = np.power(rgb_image, 1 / gamma)

    # Display color diagram
    fig, ax = plt.subplots()
    ax.imshow(rgb_corrected, extent=[0, 1, 0, 1])
    
    # Display pure spectral source
    # transform XYZ to (x,y)
    spectral_source_x = x0 / (x0 + y0 + z0)
    spectral_source_y = y0 / (x0 + y0 + z0)
    ax.plot(spectral_source_x, spectral_source_y, label='Pure Spectral Source', color='black')

    # Label axes
    plt.xlabel('x')
    plt.ylabel('y')

    fig.savefig('images/monitor_chromaticity.png')

if __name__ == "__main__":
    main()