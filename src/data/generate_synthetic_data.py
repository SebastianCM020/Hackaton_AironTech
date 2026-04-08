"""
Generador de datos sintéticos de dengue para Ecuador.
Simula ~220 cantones ecuatorianos durante 2018-2023 con correlaciones realistas.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibilidad
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Datos geográficos representativos de Ecuador (~220 cantones)
# ---------------------------------------------------------------------------
PROVINCIAS_CONFIG = {
    # (región, altitud_media_msnm, altitud_std, temp_base, densidad_pob_media)
    "Guayas":           ("costa",   20,   15, 28, 1800),
    "Manabí":           ("costa",   30,   25, 27, 600),
    "Los Ríos":         ("costa",   40,   20, 27, 700),
    "El Oro":           ("costa",   50,   30, 26, 500),
    "Esmeraldas":       ("costa",   30,   20, 27, 300),
    "Santa Elena":      ("costa",   25,   15, 26, 250),
    "Santo Domingo":    ("costa",  550,  200, 24, 900),
    "Sucumbíos":        ("amazonia", 280, 100, 26, 100),
    "Orellana":         ("amazonia", 290, 100, 26, 80),
    "Napo":             ("amazonia", 500, 200, 24, 50),
    "Pastaza":          ("amazonia", 900, 400, 22, 20),
    "Morona Santiago":  ("amazonia", 800, 300, 22, 30),
    "Zamora Chinchipe": ("amazonia", 900, 300, 22, 25),
    "Pichincha":        ("sierra",  2800, 400, 14, 1200),
    "Azuay":            ("sierra",  2600, 500, 14, 400),
    "Loja":             ("sierra",  2100, 600, 16, 300),
    "Chimborazo":       ("sierra",  2800, 400, 12, 200),
    "Tungurahua":       ("sierra",  2600, 500, 13, 500),
    "Cotopaxi":         ("sierra",  2900, 400, 11, 250),
    "Imbabura":         ("sierra",  2300, 500, 15, 350),
    "Bolívar":          ("sierra",  2500, 400, 13, 180),
    "Cañar":            ("sierra",  2700, 400, 12, 150),
    "Carchi":           ("sierra",  2900, 400, 11, 180),
}

CANTONES_POR_PROVINCIA = {
    "Guayas": ["Guayaquil", "Daule", "Milagro", "Samborondón", "Durán",
               "Naranjal", "Balzar", "Playas", "Nobol", "Palestina",
               "Pedro Carbo", "Santa Lucía", "Simón Bolívar", "Coronel Portillo",
               "El Triunfo", "General Antonio Elizalde", "Isidro Ayora",
               "Lomas de Sargentillo", "Marcelino Maridueña", "Naranjito",
               "Alfredo Baquerizo Moreno", "Colimes", "Balao"],
    "Manabí": ["Portoviejo", "Manta", "Chone", "Jipijapa", "Montecristi",
               "Bahía de Caráquez", "Calceta", "El Carmen", "Flavio Alfaro",
               "Jaramijó", "Junín", "Olmedo", "Paján", "Pedernales",
               "Pichincha", "Puerto López", "Rocafuerte", "San Vicente",
               "Santa Ana", "Sucre", "Tosagua", "24 de Mayo", "Bolívar"],
    "Los Ríos": ["Babahoyo", "Quevedo", "Buena Fé", "Valencia", "Ventanas",
                 "Vinces", "Baba", "Catarama", "Mocache", "Montalvo",
                 "Palenque", "Puebloviejo", "Urdaneta"],
    "El Oro": ["Machala", "Pasaje", "Santa Rosa", "Huaquillas", "Arenillas",
               "Atahualpa", "Balsas", "Chilla", "El Guabo", "Marcabelí",
               "Piñas", "Portovelo", "Zaruma", "Las Lajas"],
    "Esmeraldas": ["Esmeraldas", "Atacames", "Eloy Alfaro", "Muisne",
                   "Quinindé", "Rioverde", "San Lorenzo"],
    "Santa Elena": ["Santa Elena", "La Libertad", "Salinas"],
    "Santo Domingo": ["Santo Domingo", "La Concordia"],
    "Sucumbíos": ["Nueva Loja", "Cascales", "Cuyabeno", "Gonzalo Pizarro",
                  "Lago Agrio", "Putumayo", "Shushufindi", "Sucumbíos"],
    "Orellana": ["Francisco de Orellana", "Aguarico", "La Joya de los Sachas",
                 "Loreto"],
    "Napo": ["Tena", "Archidona", "El Chaco", "Quijos", "Carlos Julio Arosemena"],
    "Pastaza": ["Puyo", "Arajuno", "Mera", "Santa Clara"],
    "Morona Santiago": ["Macas", "Gualaquiza", "Huamboya", "Limón Indanza",
                        "Logroño", "Morona", "Pablo Sexto", "Palora",
                        "San Juan Bosco", "Santiago", "Sucúa", "Taisha",
                        "Tiwintza"],
    "Zamora Chinchipe": ["Zamora", "Centinela del Cóndor", "Chinchipe",
                         "El Pangui", "Nangaritza", "Palanda", "Paquisha",
                         "Yacuambi", "Yantzaza"],
    "Pichincha": ["Quito", "Cayambe", "Mejía", "Pedro Moncayo",
                  "Rumiñahui", "San Miguel de los Bancos", "Pedro Vicente Maldonado",
                  "Puerto Quito"],
    "Azuay": ["Cuenca", "Chordeleg", "Gualaceo", "Nabón", "Oña",
              "Paute", "Pucará", "San Fernando", "Santa Isabel",
              "Sevilla de Oro", "Sigsig", "El Pan", "Girón"],
    "Loja": ["Loja", "Calvas", "Catamayo", "Célica", "Chaguarpamba",
             "Espíndola", "Gonzanamá", "Macará", "Olmedo", "Paltas",
             "Pindal", "Puyango", "Quilanga", "Saraguro",
             "Sozoranga", "Zapotillo"],
    "Chimborazo": ["Riobamba", "Alausí", "Colta", "Chambo", "Chunchi",
                   "Guamote", "Guano", "Pallatanga", "Penipe", "Cumandá"],
    "Tungurahua": ["Ambato", "Baños", "Cevallos", "Mocha", "Patate",
                   "Pelileo", "Píllaro", "Quero", "Tisaleo"],
    "Cotopaxi": ["Latacunga", "La Maná", "Pangua", "Pujilí", "Salcedo",
                 "Saquisilí", "Sigchos"],
    "Imbabura": ["Ibarra", "Antonio Ante", "Cotacachi", "Otavalo",
                 "Pimampiro", "San Miguel de Urcuquí"],
    "Bolívar": ["Guaranda", "Caluma", "Chillanes", "Chimbo",
                "Echeandía", "Las Naves", "San Miguel"],
    "Cañar": ["Azogues", "Biblián", "Cañar", "Déleg", "El Tambo",
               "La Troncal", "Suscal"],
    "Carchi": ["Tulcán", "Bolívar", "Espejo", "Mira", "Montúfar", "San Pedro de Huaca"],
}


def _canton_altitude(provincia: str, canton: str) -> float:
    """Genera altitud realista para un cantón dado su provincia."""
    cfg = PROVINCIAS_CONFIG[provincia]
    alt_mean, alt_std = cfg[1], cfg[2]
    return max(0, RNG.normal(alt_mean, alt_std))


def _canton_density(provincia: str, canton: str) -> float:
    """Genera densidad poblacional para un cantón (hab/km²)."""
    cfg = PROVINCIAS_CONFIG[provincia]
    dens_mean = cfg[4]
    # Capitales de provincia tienen densidad mayor
    capitales = {p: CANTONES_POR_PROVINCIA[p][0] for p in CANTONES_POR_PROVINCIA}
    if canton == capitales.get(provincia, ""):
        factor = RNG.uniform(1.5, 2.5)
    else:
        factor = RNG.uniform(0.2, 1.2)
    return max(10, dens_mean * factor)


def _temperatura(region: str, altitud: float, semana: int) -> float:
    """Temperatura en °C según región, altitud y época del año."""
    # Componente estacional (Ecuador es tropical, variación suave)
    seasonal = 1.5 * np.sin(2 * np.pi * (semana - 10) / 52)
    # Base según altitud (lapse rate: -6.5°C/1000m)
    temp_base = 30 - (altitud / 1000) * 6.5
    temp = temp_base + seasonal + RNG.normal(0, 0.8)
    return float(np.clip(temp, 5, 38))


def _precipitacion(region: str, semana: int) -> float:
    """Precipitación en mm según región y semana epidemiológica."""
    if region == "costa":
        # Temporada lluviosa enero-mayo (semanas 1-20)
        if 1 <= semana <= 20:
            base = RNG.gamma(shape=3, scale=30)
        else:
            base = RNG.gamma(shape=1.5, scale=10)
    elif region == "amazonia":
        # Lluvia casi todo el año, pico marzo-julio
        if 10 <= semana <= 30:
            base = RNG.gamma(shape=4, scale=35)
        else:
            base = RNG.gamma(shape=3, scale=25)
    else:  # sierra
        # Dos estaciones lluviosas
        if (1 <= semana <= 15) or (35 <= semana <= 52):
            base = RNG.gamma(shape=2, scale=20)
        else:
            base = RNG.gamma(shape=1.2, scale=10)
    return float(base)


def _indice_aedes(altitud: float, temperatura: float, precipitacion: float,
                  cobertura_salud: float) -> float:
    """Índice de densidad del vector Aedes aegypti (0-1)."""
    # Aedes no sobrevive bien por encima de ~2000 msnm
    alt_factor = np.exp(-altitud / 1500)
    # Temperatura óptima 25-30°C
    temp_factor = np.exp(-((temperatura - 28) ** 2) / 50)
    # Lluvia favorece criaderos
    precip_factor = np.clip(precipitacion / 80, 0, 1)
    # Cobertura de salud reduce el índice
    control_factor = 1 - 0.4 * cobertura_salud
    index = alt_factor * temp_factor * 0.5 * (0.4 + 0.6 * precip_factor) * control_factor
    noise = RNG.normal(0, 0.05)
    return float(np.clip(index + noise, 0, 1))


def _casos_dengue(altitud: float, temperatura: float, precipitacion: float,
                  indice_aedes: float, densidad_pob: float,
                  cobertura_salud: float, semana: int, year: int) -> float:
    """Número de casos de dengue en la semana."""
    # Cero casos en alturas > 2200 msnm (sin vector)
    if altitud > 2200:
        return 0.0

    # Base risk score
    risk = (
        indice_aedes * 50
        + (temperatura - 20) * 0.8
        + precipitacion * 0.1
        + densidad_pob * 0.005
        - cobertura_salud * 15
    )
    risk = max(0, risk)

    # Tendencia temporal (incremento progresivo)
    year_trend = 1 + 0.08 * (year - 2018)

    # Componente estacional (picos en semanas 1-20 en la costa)
    seasonal = 1 + 0.6 * np.sin(2 * np.pi * (semana - 5) / 52)

    lambda_poisson = max(0.1, risk * year_trend * seasonal)
    casos = RNG.poisson(lambda_poisson)
    return float(casos)


def generate_synthetic_data() -> pd.DataFrame:
    """
    Genera el dataset sintético de dengue para Ecuador (2018-2023).

    Returns
    -------
    pd.DataFrame con ~220 cantones × 6 años × 52 semanas de registros.
    """
    records = []
    canton_id = 1

    for provincia, cantones in CANTONES_POR_PROVINCIA.items():
        cfg = PROVINCIAS_CONFIG[provincia]
        region = cfg[0]

        for canton in cantones:
            altitud = _canton_altitude(provincia, canton)
            densidad = _canton_density(provincia, canton)
            cobertura = float(np.clip(RNG.normal(0.65, 0.15), 0.2, 0.95))

            for year in range(2018, 2024):
                casos_ant = 0.0
                acumulados_mes = 0.0

                for semana in range(1, 53):
                    temp = _temperatura(region, altitud, semana)
                    precip = _precipitacion(region, semana)
                    indice = _indice_aedes(altitud, temp, precip, cobertura)
                    casos = _casos_dengue(
                        altitud, temp, precip, indice,
                        densidad, cobertura, semana, year
                    )

                    # Acumulado mensual (reinicia cada 4 semanas aprox.)
                    if semana % 4 == 1:
                        acumulados_mes = casos
                    else:
                        acumulados_mes += casos

                    # Nivel de riesgo multiclase
                    if int(casos) == 0:
                        nivel_riesgo = "bajo"
                    elif int(casos) <= 5:
                        nivel_riesgo = "medio"
                    elif int(casos) <= 20:
                        nivel_riesgo = "alto"
                    else:
                        nivel_riesgo = "muy_alto"

                    brote = int(casos > 10)

                    records.append({
                        "canton_id": canton_id,
                        "canton_name": canton,
                        "provincia": provincia,
                        "year": year,
                        "semana_epidemiologica": semana,
                        "temperatura_promedio": round(temp, 2),
                        "precipitacion_mm": round(precip, 2),
                        "casos_semana_anterior": round(casos_ant, 0),
                        "casos_acumulados_mes": round(acumulados_mes, 0),
                        "indice_aedes": round(indice, 4),
                        "altitud_msnm": round(altitud, 1),
                        "densidad_poblacional": round(densidad, 1),
                        "cobertura_salud": round(cobertura, 3),
                        "brote": brote,
                        "nivel_riesgo": nivel_riesgo,
                        "casos_dengue": round(casos, 0),
                    })

                    casos_ant = casos

            canton_id += 1

    df = pd.DataFrame(records)
    return df


def main():
    """Genera y guarda el dataset sintético."""
    output_path = Path("data/raw/dengue_ecuador_sintetico.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generando datos sintéticos de dengue para Ecuador...")
    df = generate_synthetic_data()
    df.to_csv(output_path, index=False)

    print(f"Dataset generado: {df.shape[0]:,} registros × {df.shape[1]} columnas")
    print(f"Guardado en: {output_path}")
    print("\nResumen:")
    print(f"  Cantones únicos : {df['canton_name'].nunique()}")
    print(f"  Provincias      : {df['provincia'].nunique()}")
    print(f"  Años            : {df['year'].min()} - {df['year'].max()}")
    print(f"  Total brotes    : {df['brote'].sum():,}")
    print(f"  Casos totales   : {df['casos_dengue'].sum():,.0f}")
    print("\nDistribución nivel de riesgo:")
    print(df["nivel_riesgo"].value_counts())


if __name__ == "__main__":
    main()
