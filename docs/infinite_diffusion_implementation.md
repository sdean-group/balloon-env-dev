# InfiniteDiffusion Implementation

## Overall Flow

```mermaid
flowchart LR
    Q["Request a finite<br/>space-time region"]
    L["Find overlapping<br/>output windows"]
    D["Recursively evaluate<br/>missing dependencies"]
    N["Create deterministic noise<br/>from global coordinates"]
    M["Run fixed-window<br/>diffusion model"]
    P["Pack weighted values<br/>and weights"]
    S["Sum overlapping<br/>packed windows"]
    U["Divide weighted values<br/>by summed weights"]
    C["Cache evaluated windows"]
    O["Return wind field"]

    Q --> L
    L --> D
    D --> N
    N --> M
    M --> P
    P --> S
    S --> U
    U --> O
    M --> C
    C --> D
```

## Overlap Propagation

```mermaid
flowchart TD
    subgraph T1["T = 1"]
        N1["Global coordinate noise"]
        A1["Denoising steps 0-17"]
        B1["Blend final windows"]
        N1 --> A1 --> B1
    end

    subgraph T2["T = 2"]
        N2["Global coordinate noise"]
        A2["Denoising steps 0-8"]
        B2["Blend intermediate windows"]
        C2["Denoising steps 9-17"]
        D2["Blend final windows"]
        N2 --> A2 --> B2 --> C2 --> D2
    end

    subgraph T3["T = 3"]
        N3["Global coordinate noise"]
        A3["Denoising steps 0-5"]
        B3["Blend intermediate windows"]
        C3["Denoising steps 6-11"]
        D3["Blend intermediate windows"]
        E3["Denoising steps 12-17"]
        F3["Blend final windows"]
        N3 --> A3 --> B3 --> C3 --> D3 --> E3 --> F3
    end
```

## Dependency DAG

```mermaid
flowchart BT
    Q["Requested final window"]

    F1["Final-phase window A"]
    F2["Final-phase window B"]
    F3["Final-phase window C"]

    M1["Middle-phase window 1"]
    M2["Middle-phase window 2"]
    M3["Middle-phase window 3"]
    M4["Middle-phase window 4"]
    M5["Middle-phase window 5"]

    I1["Initial window i"]
    I2["Initial window ii"]
    I3["Initial window iii"]
    I4["Initial window iv"]
    I5["Initial window v"]
    I6["Initial window vi"]

    F1 --> Q
    F2 --> Q
    F3 --> Q

    M1 --> F1
    M2 --> F1
    M2 --> F2
    M3 --> F2
    M4 --> F2
    M4 --> F3
    M5 --> F3

    I1 --> M1
    I2 --> M1
    I2 --> M2
    I3 --> M2
    I3 --> M3
    I4 --> M3
    I4 --> M4
    I5 --> M4
    I5 --> M5
    I6 --> M5
```

## Phase Construction

```text
function BUILD_INFINITE_FIELD(T, denoising_steps, split_steps):
    boundaries = [0] + split_steps + [denoising_steps]

    phase[0] = LAZY_OVERLAPPING_TENSOR(
        function(window):
            noise = COORDINATE_NOISE(
                seed,
                window.global_time_coordinates,
                window.global_y_coordinates,
                window.global_x_coordinates
            )

            state = DENOISE_SEGMENT(
                noise,
                start_step = boundaries[0],
                end_step   = boundaries[1]
            )

            weight = WINDOW_WEIGHT(window.shape)
            return PACK(weight * state, weight)
    )

    for phase_index from 1 to T - 1:
        previous_phase = phase[phase_index - 1]

        phase[phase_index] = LAZY_OVERLAPPING_TENSOR(
            dependency = previous_phase,

            function(window, overlapping_previous_windows):
                packed = SUM_OVERLAPS(overlapping_previous_windows)
                previous_state = packed.values / MAX(packed.weights, epsilon)

                state = DENOISE_SEGMENT(
                    previous_state,
                    start_step = boundaries[phase_index],
                    end_step   = boundaries[phase_index + 1]
                )

                weight = WINDOW_WEIGHT(window.shape)
                return PACK(weight * state, weight)
        )

    return phase[T - 1]
```

## Lazy Query

```text
function QUERY(field, requested_region):
    required_final_windows = WINDOWS_COVERING(requested_region)

    for window in required_final_windows:
        EVALUATE_RECURSIVELY(window)

    packed_result = SUM_OVERLAPS(
        cached final windows intersecting requested_region
    )

    normalized_wind =
        packed_result.values / MAX(packed_result.weights, epsilon)

    return DENORMALIZE_AND_SPLIT_UV(normalized_wind)
```

## Recursive Evaluation and Caching

```text
function EVALUATE_RECURSIVELY(window):
    if CACHE_CONTAINS(window):
        return CACHE_GET(window)

    dependencies = PARENT_WINDOWS_REQUIRED_BY(window)

    parent_values = []
    for parent in dependencies:
        parent_values.append(EVALUATE_RECURSIVELY(parent))

    value = window.phase_function(parent_values)
    CACHE_PUT(window, value)
    return value
```
