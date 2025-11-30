# Game Agent

Agente de IA para analise de games usando screenshots e visao computacional com LangGraph e Claude via OpenRouter.

## Caracteristicas

- **Screenshot automatico**: Captura a tela do jogo automaticamente
- **Analise visual com IA**: Usa Claude Vision para interpretar imagens do jogo
- **Comandos em linguagem natural**: Fale naturalmente com o agente
- **LangGraph**: Workflow inteligente com multiplas etapas
- **OpenRouter**: Suporte a multiplos modelos de IA
- **Text-to-Speech**: Vozes naturais do Microsoft Edge (pt-BR)

## Instalacao

### Pre-requisitos

- Python 3.11+
- uv (gerenciador de pacotes)

### Setup

1. Clone ou navegue ate o diretorio do projeto:

```bash
cd game-agent
```

2. Instale as dependencias com uv:

```bash
uv sync
```

3. Configure sua API key do OpenRouter:

```bash
cp .env.example .env
# Edite o .env e adicione:
# OPENROUTER_API_KEY=sk-or-v1-...
# ENABLE_TTS=true  (opcional, para ativar voz)
```

4. (Opcional) Para audio funcionar, instale um player:

```bash
# Linux
sudo apt install mpg123
# ou
sudo apt install mpv

# Mac
brew install mpg123

# Windows: funciona nativamente
```

## Uso

### Modo Interativo

Execute o agente em modo interativo:

```bash
uv run game-agent
```

### Exemplos de Comandos

Uma vez iniciado, voce pode usar comandos como:

**Analise de jogos:**
- `"analise essa etapa do game para mim"` - Tira screenshot e analisa a tela
- `"leia e interprete esse texto para mim"` - Tira screenshot e le texto na tela
- `"o que voce ve na tela do jogo?"` - Analise geral da tela
- `"quais sao os objetivos visiveis?"` - Identifica objetivos no HUD
- `"me ajude com essa missao"` - Analise estrategica

**Controles:**
- `tts on` - Ativa a voz (text-to-speech)
- `tts off` - Desativa a voz
- `quit` ou `exit` - Sair do programa

## Estrutura do Projeto

```
game-agent/
├── src/
│   └── game_agent/
│       ├── __init__.py       # Entry point e CLI
│       ├── agent.py          # LangGraph agent implementation
│       └── tools.py          # Screenshot tools
├── screenshots/              # Screenshots salvos automaticamente
├── pyproject.toml           # Configuracao do projeto
├── .env.example             # Exemplo de variaveis de ambiente
└── README.md                # Este arquivo
```

## Como Funciona

1. **Voce faz uma pergunta**: "analise essa etapa do game"
2. **O agente decide**: Usa LangGraph para decidir que precisa de um screenshot
3. **Tira screenshot**: Usa a ferramenta `take_screenshot` para capturar a tela
4. **Analisa com Vision**: Envia a imagem para Claude via OpenRouter
5. **Responde**: Fornece insights detalhados sobre o que ve

## Ferramentas Disponiveis

### `take_screenshot(save_path: Optional[str])`

Captura a tela inteira e retorna:
- Caminho do arquivo salvo
- Imagem em base64 para analise

### `take_region_screenshot(x, y, width, height, save_path: Optional[str])`

Captura uma regiao especifica da tela.

## Tecnologias

- **LangGraph**: Orquestracao de workflow com grafos
- **LangChain**: Framework para aplicacoes LLM
- **OpenRouter**: Gateway para multiplos modelos de IA
- **Claude (Anthropic)**: Modelo de linguagem com visao
- **MSS**: Captura de tela multiplataforma
- **Pillow**: Processamento de imagens

## Desenvolvimento

Para modificar ou estender o agente:

1. Edite `tools.py` para adicionar novas ferramentas
2. Edite `agent.py` para modificar o workflow do LangGraph
3. Teste com `uv run game-agent`

## Licenca

MIT
