FFHQ_PROMPT = "a sharp photo of a face"  # Table 6's caption

METFACES_PROMPT = "a portrait of a face"
METFACES_PHOTO_PROMPT = "a realistic sharp photo of a face"
SNGFACES_PROMPT = "an oil painting portrait of a face"
MEDICAL_PROMPT = "a chest x-ray"
DOG_PROMPT = "a sharp photo of a dog"
CAT_PROMPT = "a sharp photo of a cat"
# WRONG_PROMPT = "a sharp photo of a cat"  # Table 9's deliberately-wrong prompt for a dog image

# Table 6's parameters
GAUSS = ["problem.sigma_kernel=6", "problem.sigma_y=0.01"]
MOTION = ["problem.sigma_y=0.01"]
SR16 = ["problem.downscaling_factor=16", "problem.sigma_y=0.01"]
BOX = ["problem.mask_size=512", "problem.sigma_y=0.01"]

# New Degradations
DISK = ["problem.radius=9", "problem.sigma_y=0.01"]
ANISO = ["problem.sigma_major=9", "problem.sigma_minor=1.5", "problem.angle=35",
         "problem.sigma_y=0.01"]
