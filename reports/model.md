# сверточная модель (CNN) для определения класса направления взгляда

```markdown
параметры: ~2.9M
файл: `gaze-classification.pth`

архитектура:

Input (3, 224, 224)
↓
Conv Block 1: 3→32 filters
Conv Block 2: 32→64 filters
Conv Block 3: 64→128 filters
Conv Block 4: 128→256 filters
↓
Flatten → FC(4096→512) → Dropout(0.4) → FC(512→5)
↓
Output (5 классов)
```

**параметры:** ~1.5M

## обучение

**файл:** `train.py`

| параметр | значение |
|----------|----------|
| optimizer | Adam |
| learning rate | 0.0001 |
| loss | CrossEntropyLoss |
| batch size | 128 |
| epochs | 10 |
| device | CUDA/CPU (auto) |

## результаты

- **val accuracy:** ~99%
- **размер модели:** ~41 MB
- **inference time (GPU):** ~5-10ms

## особенности

- BatchNorm для стабильности
- Dropout(0.4) против переобучения
- AdaptiveAvgPool для фиксированного выхода