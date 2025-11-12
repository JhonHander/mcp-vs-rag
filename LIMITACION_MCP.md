# 📝 Implementación de Límites para Contexto MCP

## 🎯 Problema Resuelto

Anteriormente, las búsquedas MCP (Tavily y DuckDuckGo) devolvían demasiado contexto, causando:
- ❌ Alto consumo de tokens (2091-3275 tokens en los experimentos)
- ❌ Costos elevados ($0.023-$0.035 por experimento MCP vs $0.007-$0.012 para RAG)
- ❌ Riesgo de exceder límites de contexto del LLM

## ✅ Solución Implementada

### 1. **Archivo de Configuración Centralizado**
**`src/config/mcp_config.py`**

```python
MCP_SEARCH_LIMITS = {
    "tavily": {
        "max_results": 3,        # Reducido de ~5-10 por defecto
        "search_depth": "basic"  # "basic" en lugar de "advanced"
    },
    "duckduckgo": {
        "max_results": 3         # Reducido de ~10 por defecto
    }
}

MAX_CONTEXT_LENGTH = 3000  # ~750 tokens aproximadamente
```

### 2. **Búsqueda MCP Limitada**
**`src/workflow/main_workflow.py` - función `search_mcp_context()`**

Ahora la búsqueda:
1. ✅ Aplica límites configurables por servidor (`max_results`, `search_depth`)
2. ✅ Trunca contexto si excede 3000 caracteres
3. ✅ Registra cuando hay truncamiento para análisis

**Antes:**
```python
result = await search_tool.ainvoke({"query": state["prompt"]})
return {"mcp_context": str(result)}
```

**Después:**
```python
base_params = {"query": state["prompt"]}
server_config = get_mcp_search_config(state["mcp_server"])
search_params = {**base_params, **server_config}

result = await search_tool.ainvoke(search_params)
was_truncated, final_context = should_truncate_context(str(result))
return {"mcp_context": final_context}
```

## 📊 Impacto Esperado

### Reducción de Tokens Estimada:

| Métrica | Antes | Después | Reducción |
|---------|-------|---------|-----------|
| **Tokens MCP (promedio)** | 1,000-1,500 | 400-700 | ~50-60% |
| **Costo MCP por experimento** | $0.024-$0.035 | $0.010-$0.018 | ~50-60% |
| **Context length** | Ilimitado | Max 3,000 chars | Controlado |

### Ejemplo Real (Config 2: GPT-5 + DuckDuckGo):
- **Antes**: 1,398 input tokens → $0.007 input + $0.028 output = **$0.035 total**
- **Después (estimado)**: ~600 input tokens → $0.003 input + $0.015 output = **~$0.018 total**

## 🎚️ Configuración Ajustable

Puedes modificar los límites en `src/config/mcp_config.py`:

```python
# Para resultados más completos (más tokens, más costo):
"max_results": 5,
"search_depth": "advanced"  # Solo Tavily
MAX_CONTEXT_LENGTH = 5000

# Para resultados más concisos (menos tokens, menos costo):
"max_results": 2,
"search_depth": "basic"
MAX_CONTEXT_LENGTH = 2000
```

## 📁 Archivos Modificados

1. ✅ **`src/config/mcp_config.py`** (NUEVO)
   - Configuración centralizada de límites MCP
   - Funciones helper: `get_mcp_search_config()`, `should_truncate_context()`

2. ✅ **`src/config/__init__.py`** (NUEVO)
   - Módulo de configuración

3. ✅ **`src/workflow/main_workflow.py`** (MODIFICADO)
   - Importa configuración MCP
   - Aplica límites en `search_mcp_context()`
   - Registra truncamientos

## 🚀 Próximos Pasos

1. **Probar los nuevos límites**:
   ```bash
   python run_experiment.py 1
   ```

2. **Verificar reducción de costos**:
   - Revisar `cost_summary` en el JSON de salida
   - Comparar con resultados anteriores

3. **Ajustar si es necesario**:
   - Si las respuestas MCP pierden calidad → aumentar `max_results` a 4-5
   - Si los costos siguen altos → reducir `MAX_CONTEXT_LENGTH` a 2000

## 🔍 Monitoreo

El sistema ahora imprime logs durante la ejecución:

```
🔍 MCP Search (tavily): {'query': '...', 'max_results': 3, 'search_depth': 'basic'}
⚠️  Context truncated from 4521 to 3000 chars
```

Esto te ayuda a ver cuándo y cuánto se está truncando el contexto.

## 📌 Notas Importantes

- ✅ **No afecta RAG**: Los límites solo aplican a búsquedas MCP
- ✅ **No afecta RAGAS**: La evaluación no cambió
- ✅ **Tracking de costos intacto**: Sigue funcionando normalmente
- ✅ **Retrocompatible**: Si no existe config, usa valores por defecto seguros

## 💡 Recomendaciones

**Para producción:**
- Tavily: `max_results=3`, `search_depth="basic"` (balance calidad/costo)
- DuckDuckGo: `max_results=3` (más económico)
- MAX_CONTEXT_LENGTH: 3000 (suficiente para respuestas completas)

**Para máxima calidad (experimentación):**
- Tavily: `max_results=5`, `search_depth="advanced"`
- DuckDuckGo: `max_results=5`
- MAX_CONTEXT_LENGTH: 5000

**Para mínimo costo:**
- Tavily: `max_results=2`, `search_depth="basic"`
- DuckDuckGo: `max_results=2`
- MAX_CONTEXT_LENGTH: 2000
