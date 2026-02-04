# документация на API

**URL:** `http://localhost:5252`

## endpoints

### `GET /health`
проверка статуса жизни

- 200:
```json
{"status": "online", "model_device": "cuda", "classes": [...]}
```

### `POST /predict`
предсказание направления взгляда

пример запроса:

```curl
curl -X POST http://localhost:5252/predict -F "file=@image.jpg"
```

- 200:

```json
{
  "success": true,
  "predicted_class": "straight",
  "confidence": 0.92
}
```

### `GET /available-classes`
Список классов: `down`, `left`, `right`, `straight`, `up`

## Коды ответов
- **200** - окей
- **400** - неверный формат
- **500** - ошибка сервера