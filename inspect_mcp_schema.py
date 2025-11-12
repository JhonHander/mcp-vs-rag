"""
Script para inspeccionar los parámetros que aceptan las herramientas MCP.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mcp.mcp_client import get_search_tool_for_server


async def inspect_tool_schema(server_type: str):
    """Inspecciona el schema de una herramienta MCP."""
    print(f"\n{'=' * 70}")
    print(f"Inspeccionando {server_type.upper()}")
    print('=' * 70)
    
    try:
        tool = await get_search_tool_for_server(server_type)
        
        if not tool:
            print(f"No se pudo obtener la herramienta para {server_type}")
            return
        
        print(f"\n📛 Nombre: {tool.name}")
        print(f"\n📝 Descripción:\n{tool.description}")
        print(f"\n⚙️  Schema de argumentos:")
        print("-" * 70)
        
        # El schema está en tool.args que es un dict con la estructura JSON Schema
        if hasattr(tool, 'args') and tool.args:
            import json
            print(json.dumps(tool.args, indent=2))
        else:
            print("No hay schema de argumentos disponible")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    await inspect_tool_schema("tavily")
    await inspect_tool_schema("duckduckgo")


if __name__ == "__main__":
    asyncio.run(main())
