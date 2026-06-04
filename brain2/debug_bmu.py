import brain2
b = brain2.Brain(8, 8, 16)
ckpt = "checkpoints/stage5_math"
b.load_components(
    predictor_path=f"{ckpt}/predictor.bin",
    language_path=f"{ckpt}/language.bin",
    som_path=f"{ckpt}/som.bin",
    episodic_path=f"{ckpt}/episodic.bin"
)

v_pranay = b.language.encode("pranay")
v_941 = b.language.encode("941")
print("BMU pranay:", b.som.find_bmu(v_pranay))
print("BMU 941:", b.som.find_bmu(v_941))
