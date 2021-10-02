
# Hur man använder simulerings- och analysverktyget för stötlabben

## Simuleringar

Alla filer som behövs ska sparas i en mapp, t.ex. "simulation_configs". I den mappen ska det finnas två mappar, `configs` och `objects`.

Exempel på en filuppsättning:

```txt
simulation_configs/
├── configs/
│   ├── collision.json
│   ├── spin1.json
│   └── spin2.json
└── objects/
    ├── puck1.json
    └── puck2.json
```

### Objects

I denna mappen sparas filer för alla objekt som används i laborationen, i json-format. Exempel `puck1.json`:

```json
{
    "radius": 50,
    "markers": [
        [0, 40],
        [40, 0],
        [-40, 0]
    ],
    "mass": 0.2,
    "friction_coeff": 0.01
}
```

* `radius` är radien på objektet mätt i millimeter.
* `markers` är en lista med punkter som representerar objektets markörer.
* `mass` är objektets massa mätt i kilogram.
* `friction_coeff` är objektets friktionskoefficient mot golvet.

### Configs

I denna mappen finns alla labbuppställningar, även dem i json-format. Exempel `collision.json`:

```json
{
    "timespan": 10.0,
    "dt": 0.01,
    "noise_level": 1.0,
    "elasticity_coeff": 1.0,
    "collision_friction_coeff": 0.5,
    "rigidbodies": [
        {
            "object": "puck1",
            "position": [-200, 0],
            "velocity": [200, -50],
            "theta": 1,
            "omega": 3
        },
        {
            "object": "puck2",
            "position": [200, 0],
            "velocity": [-100, 0],
            "theta": -0.5,
            "omega": 2
        }
    ]
}
```

* `timespan` är hur länge simuleringen ska köras.
* `dt` är hur ofta simuleringen ska köras.
* `noise_level` är hur stort bruset är mätt i millimeter.
* `rigidbodies` är en lista med objekt som ska simuleras.
* `object` är namnet på objektet i `objects`-mappen som ska simuleras.
* `position` är startpositionen på objektet i millimeter.
* `velocity` är starthastigheten på objektet i millimeter per sekund.
* `theta` är startvinkeln på objektet i radianer.
* `omega` är startrotationshastigheten på objektet i radianer per sekund.
