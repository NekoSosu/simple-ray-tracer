import taichi as ti
import taichi.math as tm
from taichi.math import vec3, reflect, refract, sqrt, pow, sin, cos, tan, normalize, dot, cross

# Initialize Taichi
ti.init(arch=ti.gpu)

# Constants
APERTURE = 1.0
N_SPHERES = 10
RESOLUTION = (640, 480)
N_SUBPIXEL = 4
MAX_DISTANCE = 1e10
MAX_DEPTH = 15
EPSILON = 0.022
GAMMA = 1.8
CONVERGENCE_FACTOR = 1.0
RUSSIAN_ROULETTE_PROB = 0.8  # Probability of continuing the path trace

# Fields
pixels = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
accum = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)
corrected = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)

# Classes
@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

@ti.dataclass        
class Camera:
    position: vec3
    look_at: vec3
    fov: ti.f32
    focal_length: ti.f32 
    aperture: ti.f32

@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32
    emission: vec3
    colour: vec3
    material: ti.i32  # 1: DIFF&SPEC, 2: REFR
    roughness: ti.f32
    refract_idx: ti.f32

    @ti.func
    def intersect(self, ray: Ray) -> ti.f32:
        oc = self.center - ray.origin
        half_b = dot(oc, ray.direction)
        c = dot(oc, oc) - self.radius * self.radius
        discriminant = half_b * half_b - c

        t = 0.0
        if discriminant >= 0:
            t = half_b - sqrt(discriminant)
        if t <= EPSILON:
            t = half_b + sqrt(discriminant)
        if t <= EPSILON:
            t = 0.0
        return t

# Scene setup
spheres = Sphere.field(shape=(N_SPHERES,))

@ti.func
def intersect_scene(ray: Ray):
    """Find the nearest intersection of a ray with all spheres in the scene."""
    id = -1
    t = MAX_DISTANCE
    for i in ti.static(range(N_SPHERES)):
        d = spheres[i].intersect(ray)
        if d > 0 and d < t:
            id = i
            t = d
    return id, t

# Sampling functions
@ti.func
def sample_uniform_hemisphere(n: vec3):
    basis_y = n
    basis_x = vec3(1, 0, 0) if ti.abs(n[0]) <= 0.1 else vec3(0, 1, 0)
    basis_x = normalize(cross(basis_x, basis_y))
    basis_z = cross(basis_x, basis_y)

    y = ti.random(ti.f32)
    r = sqrt(1 - y*y)
    phi = 2 * tm.pi * ti.random(ti.f32)
    x, z = r * cos(phi), r * sin(phi)

    return normalize(x * basis_x + y * basis_y + z * basis_z)

@ti.func
def sample_cosine_weighted_hemisphere(n: vec3):
    basis_y = n
    basis_x = vec3(1, 0, 0) if ti.abs(n[0]) <= 0.1 else vec3(0, 1, 0)
    basis_x = normalize(cross(basis_x, basis_y))
    basis_z = cross(basis_x, basis_y)

    r = sqrt(ti.random(ti.f32))
    theta = 2 * tm.pi * ti.random(ti.f32)
    x, z = r * cos(theta), r * sin(theta)
    y = sqrt(1 - r*r)
  
    return normalize(x * basis_x + y * basis_y + z * basis_z)

@ti.func
def sample_uniform_disk():
    r = sqrt(ti.random(ti.f32))
    theta = 2 * tm.pi * ti.random(ti.f32)
    return r * cos(theta), r * sin(theta)

@ti.func
def schlick_approximation(cos_theta: ti.f32, eta: ti.f32) -> ti.f32:
    r0 = (1 - eta) / (1 + eta)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cos_theta), 5)

# Path tracing
@ti.func
def trace_path(ray: Ray) -> vec3:
    colour = vec3(0.0, 0.0, 0.0)
    absorption = vec3(1.0, 1.0, 1.0)
    origin, direction = ray.origin, normalize(ray.direction)

    for depth in range(MAX_DEPTH):
        intersect_id, t = intersect_scene(Ray(origin, direction))
        if intersect_id < 0:
            colour += absorption * vec3(0.2, 0.2, 0.2)
            break

        sphere = spheres[intersect_id]
        hit_point = origin + t * direction
        hit_normal = normalize(hit_point - sphere.center)

        colour += absorption * sphere.emission
        absorption *= sphere.colour

        # Russian Roulette termination
        if depth > 3:
            rr_prob = min(absorption.max(), RUSSIAN_ROULETTE_PROB)
            if ti.random(ti.f32) >= rr_prob:
                break
            absorption /= rr_prob

        if sphere.material == 1:  # Diffuse & Specular
            cos_direction = sample_cosine_weighted_hemisphere(hit_normal)
            specular_direction = reflect(direction, hit_normal)
            direction = normalize(tm.mix(specular_direction, cos_direction, pow(sphere.roughness, 2)))
        else:  # Refractive
            cos_theta = dot(-direction, hit_normal)
            entering = cos_theta > 0
            normal = hit_normal if entering else -hit_normal
            eta = 1 / sphere.refract_idx if entering else sphere.refract_idx

            sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            cannot_refract = eta * sin_theta > 1.0

            reflectance = schlick_approximation(abs(cos_theta), eta)

            if cannot_refract or ti.random(ti.f32) < reflectance:
                cos_direction = sample_cosine_weighted_hemisphere(normal)
                specular_direction = reflect(direction, hit_normal)
                direction = normalize(tm.mix(specular_direction, cos_direction, pow(sphere.roughness, 2)))
            else:
                cos_direction = sample_cosine_weighted_hemisphere(-normal)
                refraction_direction = normalize(refract(direction, normal, eta))
                direction = normalize(tm.mix(refraction_direction, cos_direction, pow(sphere.roughness, 2)))

        origin = hit_point

    return colour

# Rendering
@ti.kernel
def render(camera: Camera, n: ti.i32):
    aspect_ratio = RESOLUTION[0] / RESOLUTION[1]
    half_height = tan(tm.radians(camera.fov) / 2) * camera.focal_length
    half_width = half_height * aspect_ratio
    
    w = normalize(camera.position - camera.look_at)
    u = normalize(cross(vec3(0, 1, 0), w))
    v = cross(w, u)

    lower_left = camera.position - camera.focal_length*w - half_height*v - half_width*u
    du, dv = 2 * half_width * u / RESOLUTION[0], 2 * half_height * v / RESOLUTION[1]

    inv_n = CONVERGENCE_FACTOR / ti.cast(n, ti.f32)
    for i, j in pixels:
        accum[i, j] = vec3(0, 0, 0)
        for _ in range(N_SUBPIXEL):  # subpixel sampling
            jitter_x, jitter_y = ti.random(ti.f32), ti.random(ti.f32)
            offset_x, offset_y = sample_uniform_disk()
            ray_origin = camera.position + offset_x*camera.aperture*u + offset_y*camera.aperture*v
            ray_direction = lower_left + (i + jitter_x)*du + (j + jitter_y)*dv - ray_origin
            accum[i, j] += trace_path(Ray(ray_origin, ray_direction))/N_SUBPIXEL

        pixels[i, j] += (accum[i, j] - pixels[i, j]) * inv_n
        pixels[i, j] = tm.clamp(pixels[i, j], 0.0, 1.0)

@ti.kernel
def apply_gamma_correction():
    for i, j in pixels:
        corrected[i, j] = pow(pixels[i, j], 1 / GAMMA)

# Scene initialization
def init_scene():
    spheres[0] = Sphere(vec3(-1e5 + 1, 40.8, 81.6), 1e5, vec3(0), vec3(0.75, 0.25, 0.25), 1, 1.0, 0)  # Left
    spheres[1] = Sphere(vec3(1e5 + 99, 40.8, 81.6), 1e5, vec3(0), vec3(0.25, 0.25, 0.75), 1, 1.0, 0)  # Right
    spheres[2] = Sphere(vec3(50, 40.8, -1e5), 1e5, vec3(0), vec3(0.75), 1, 0.4, 0)  # Back
    spheres[3] = Sphere(vec3(50, 40.8, 1e5 + 170), 1e5, vec3(0), vec3(0.15), 1, 1.0, 0)  # Front
    spheres[4] = Sphere(vec3(50, -1e5, 81.6), 1e5, vec3(0), vec3(0.75), 1, 1.0, 0)  # Bottom
    spheres[5] = Sphere(vec3(50, 1e5 + 81.6, 81.6), 1e5, vec3(0), vec3(0.75), 1, 1.0, 0)  # Top
    spheres[6] = Sphere(vec3(27, 16.5, 47), 16.5, vec3(0), vec3(1), 1, 0.0, 0)  # Mirror
    spheres[7] = Sphere(vec3(73, 16.5, 78), 16.5, vec3(0), vec3(0.7, 1, 0.9), 2, 0.0, 1.55)  # Glass
    spheres[8] = Sphere(vec3(50, 681.6 - 0.27, 81.6), 600, vec3(12,12,12), vec3(1), 1, 1.0, 0)  # Light
    spheres[9] = Sphere(vec3(80,55,45), 5, vec3(14,10,4), vec3(1), 1, 1.0, 0)  # Light bulb

# Main rendering loop
def main():
    init_scene()
    camera = Camera(vec3(50, 40.8, 169), vec3(50, 40, 81.6), 60, 96, APERTURE)
    gui = ti.GUI("Pathtracer", res=RESOLUTION)

    n = 0
    while gui.running:
        render(camera, n)
        apply_gamma_correction()
        n += 1
        gui.set_image(corrected)
        gui.show()

if __name__ == "__main__":
    main()