"""Textos de la interfaz de RAG_SA en español y portugués.

No modifica las entidades, los datos RDF ni los nombres propios de las lenguas.
"""

LANGUAGES = {"es": "Español", "pt": "Português"}

QUESTIONS = {
    "es": [
        "¿Qué sabes sobre la lengua kariri?",
        "¿Qué lenguas indígenas se hablan en Perú?",
        "¿A qué familia pertenece el jaqaru?",
        "¿Qué lenguas pertenecen a la familia arawak?",
        "¿En qué países se hablan lenguas de la familia quechua?",
        "¿Qué lenguas se hablan en Bolivia?",
        "¿Qué lenguas aparecen asociadas a Brasil?",
        "¿Qué lenguas pertenecen a la familia tupí-guaraní?",
        "¿En qué país se habla el matsés?",
        "¿Qué sabemos sobre el asháninka?",
        "¿Cuáles son algunas lenguas con pocos hablantes?",
    ],
    "pt": [
        "O que você sabe sobre a língua kariri?",
        "Quais línguas indígenas são faladas no Peru?",
        "A que família pertence o jaqaru?",
        "Quais línguas pertencem à família arawak?",
        "Em quais países são faladas línguas da família quéchua?",
        "Quais línguas são faladas na Bolívia?",
        "Quais línguas estão associadas ao Brasil?",
        "Quais línguas pertencem à família tupi-guarani?",
        "Em qual país se fala matsés?",
        "O que sabemos sobre o asháninka?",
        "Quais são algumas línguas com poucos falantes?",
    ],
}

STRINGS = {
 "es": {
  "page_title": "Lenguas Indígenas · Sudamérica",
  "header": "Lenguas indígenas · Sudamérica",
  "subtitle": "GraphRAG Explorer",
  "intro": "Este proyecto permite conversar con datos lingüísticos especializados sobre las lenguas de Sudamérica. Puedes preguntar por una lengua, explorar sus agrupaciones y consultar las relaciones y los países registrados en el grafo.",
  "tab_chat": "💬 Consultas", "tab_explore": "🔎 Explorar", "tab_about": "ℹ️ Acerca de",
  "mode": "Modo de pregunta", "suggested_mode": "📋 Preguntas sugeridas", "free_mode": "✍️ Pregunta libre",
  "groq_ready": "🟢 Asistente disponible", "model_info": "Modelo en uso: {model}",
  "groq_missing": "Para conversar con el asistente, configura tu clave GROQ_API_KEY en los secretos de Streamlit.",
  "advanced": "⚙️ Opciones y conexión", "semantic": "Búsqueda semántica ampliada (opcional)",
  "secret_info": "La clave es privada y no aparece en el historial.", "check_groq": "Comprobar conexión con Groq",
  "check_success": "¡Todo listo! Groq está conectado y el modelo {model} está disponible.",
  "check_unavailable": "La conexión funciona, pero el modelo {model} no está disponible para esta cuenta. Puedes cambiar GROQ_MODEL.",
  "question_select": "Selecciona una pregunta", "question_placeholder": "Elige una pregunta para empezar…",
  "question_free": "Escribe tu pregunta aquí", "question_or_write": "O escribe tu propia pregunta (tendrá prioridad sobre la seleccionada).", "question_free_placeholder": "Escribe aquí lo que te gustaría saber…",
  "ask": "✨ Preguntar", "loading": "Buscando información para responderte…",
  "preparing": "Preparando las relaciones entre lenguas…", "missing_ttl": "Falta el archivo `grafo_enriquecido.ttl` en la carpeta del proyecto.",
  "load_error": "No se pudieron abrir los datos del proyecto.", "load_error_detail": "Detalle: {name}. Revisa los archivos de datos.",
  "unexpected_error": "No pude procesar esta pregunta. ¿Quieres intentarlo otra vez?",
  "unexpected_detail": "Error interno: {name}.", "choose_first": "Elige o escribe una pregunta para continuar.",
  "user": "Tú", "assistant": "Asistente", "groq_error": "No pude conectar con Groq. Si quieres, revisa el detalle más abajo.",
  "evidence": "🔎 Relaciones consultadas", "clear": "🗑️ Borrar conversación",
  "empty_chat": "¿Por dónde empezamos? Elige una pregunta o escribe la tuya.",
  "evidence_empty": "No encontré relaciones suficientes para esta pregunta.",
  "type_language": "Lengua", "type_family": "Familia o subgrupo",
  "families_related": "Familias y subgrupos", "more_info": "Para conocer más",
  "source": "Enlace {n}", "family_meta": "{n} lenguas asociadas",
  "card_language": "En este proyecto, esta lengua o variedad aparece {location}{family}. Puedes preguntarle al asistente para saber más.",
  "card_family": "Esta agrupación reúne {n} lenguas o variedades registradas en el grafo. Puedes explorar sus integrantes.",
  "card_location": "vinculada a {countries}", "card_no_location": "con una ficha propia", "card_family_suffix": " y relacionada con {families}",
  "family_label": "Familia(s)",
  "kind": "Tipo", "languages": "Lenguas", "families": "Familias",
  "country": "País", "all_countries": "Todos", "family_filter": "Familia lingüística", "all_families": "Todas",
  "filter": "Buscar por nombre o ISO", "filter_placeholder": "Quechua, kariri, jaqaru…",
  "results": "{n} resultados · se muestran hasta 50", "download": "⬇️ Descargar selección en CSV",
  "metric_languages": "Lenguas y variedades", "metric_families": "Familias y subgrupos", "metric_described": "Lenguas con descripción",
  "about_project_title": "El proyecto",
  "about_project": "Este proyecto surge como una forma de conversar con datos lingüísticos especializados. Parte especialmente de los registros y clasificaciones de Glottolog, organizados en un grafo de conocimiento, para hacer preguntas en lenguaje natural y explorar relaciones entre lenguas, agrupaciones y países. Es una manera adicional de consultar la información disponible.",
  "about_method_title": "Cómo funciona",
  "about_method": "La aplicación utiliza un grafo de conocimiento para organizar información sobre lenguas y sus relaciones. Cuando haces una pregunta, el sistema busca datos relevantes en el grafo y utiliza un modelo de lenguaje para elaborar una respuesta en español o portugués.",
  "about_sources_title": "Fuentes de información",
  "about_sources": "El grafo combina registros y clasificaciones de Glottolog con información de Wikidata y descripciones de Wikipedia. Algunos textos también contienen referencias a Ethnologue; no se trata de una importación independiente de esa base de datos. Cada recurso aporta información diferente.",
  "about_technical_title": "Arquitectura técnica",
  "about_technical": "Los datos se organizan mediante RDF (Resource Description Framework). La recuperación combina relaciones explícitas del grafo y búsqueda textual con BM25, con una opción de búsqueda semántica mediante embeddings multilingües. Las respuestas se generan utilizando un modelo de lenguaje a través de la API de Groq.",
  "about_scope_title": "Alcance",
  "about_scope": "El grafo no es un inventario completo de las lenguas indígenas de Sudamérica. Algunos datos pueden faltar o estar desactualizados, y las estimaciones del número de hablantes deben interpretarse con cautela.",
  "chat_section_title": "Consultar el grafo",
  "chat_section_help": "Selecciona una pregunta o escribe la tuya.",
  "explore_intro": "Explora los registros paso a paso: elige un país, una agrupación y, si quieres, un subgrupo o una lengua. También puedes buscar directamente por nombre o código ISO.",
  "explore_country_help": "Selecciona un país para ver las agrupaciones y lenguas asociadas.",
  "explore_country_placeholder": "Selecciona un país…", "explore_group": "Agrupación principal",
  "explore_group_placeholder": "Todas las agrupaciones del país", "explore_subgroup": "Subgrupo (opcional)",
  "explore_subgroup_placeholder": "Todos los subgrupos", "explore_deeper": "Grupo más específico (opcional)",
  "explore_language": "Lengua o variedad", "explore_language_placeholder": "Selecciona una lengua…",
  "explore_search_help": "Buscar directamente por nombre o código ISO",
  "explore_no_matches": "No hay lenguas asociadas a esta combinación.",
  "explore_count": "{n} lenguas o variedades en esta selección.",
  "explore_select": "Elige una lengua para ver su ficha, o utiliza el buscador.",
  "explore_no_country": "Elige un país para comenzar, o escribe el nombre de una lengua en el buscador.",
  "explore_ungrouped": "Sin agrupación registrada ({n})",
  "explore_groups_count": "{n} agrupaciones principales para este país.",
  "about_responsible_title": "Responsable y sugerencias",
  "about_responsible": "Responsable: Javier Maximiliano Vera Zúñiga. Puedes escribir para compartir comentarios o sugerencias sobre el proyecto.",
  "source_links_label": "Recursos y enlaces",
  "footer": "Lenguas indígenas · Sudamérica"
 },
 "pt": {
  "page_title": "Línguas indígenas · América do Sul",
  "header": "Línguas indígenas · América do Sul",
  "subtitle": "GraphRAG Explorer",
  "intro": "Este projeto permite conversar com dados linguísticos especializados sobre as línguas da América do Sul. Você pode perguntar sobre uma língua, explorar seus agrupamentos e consultar as relações e os países registrados no grafo.",
  "tab_chat": "💬 Perguntas", "tab_explore": "🔎 Explorar", "tab_about": "ℹ️ Sobre o projeto",
  "mode": "Modo de pergunta", "suggested_mode": "📋 Perguntas sugeridas", "free_mode": "✍️ Pergunta livre",
  "groq_ready": "🟢 Assistente disponível", "model_info": "Modelo em uso: {model}",
  "groq_missing": "Para conversar com o assistente, configure sua chave GROQ_API_KEY nos segredos do Streamlit.",
  "advanced": "⚙️ Opções e conexão", "semantic": "Busca semântica ampliada (opcional)",
  "secret_info": "A chave é privada e não aparece no histórico.", "check_groq": "Verificar conexão com o Groq",
  "check_success": "Tudo pronto! O Groq está conectado e o modelo {model} está disponível.",
  "check_unavailable": "A conexão funciona, mas o modelo {model} não está disponível nesta conta. Você pode alterar GROQ_MODEL.",
  "question_select": "Selecione uma pergunta", "question_placeholder": "Escolha uma pergunta para começar…",
  "question_free": "Escreva sua pergunta aqui", "question_or_write": "Ou escreva sua própria pergunta (ela terá prioridade sobre a selecionada).", "question_free_placeholder": "Digite aqui o que você gostaria de saber…",
  "ask": "✨ Perguntar", "loading": "Buscando informações para responder…",
  "preparing": "Preparando as relações entre línguas…", "missing_ttl": "O arquivo `grafo_enriquecido.ttl` não está na pasta do projeto.",
  "load_error": "Não foi possível abrir os dados do projeto.", "load_error_detail": "Detalhe: {name}. Confira os arquivos de dados.",
  "unexpected_error": "Não consegui processar essa pergunta. Quer tentar novamente?",
  "unexpected_detail": "Erro interno: {name}.", "choose_first": "Escolha ou escreva uma pergunta para continuar.",
  "user": "Você", "assistant": "Assistente", "groq_error": "Não consegui conectar ao Groq. Se quiser, veja os detalhes abaixo.",
  "evidence": "🔎 Relações consultadas", "clear": "🗑️ Limpar conversa",
  "empty_chat": "Por onde começamos? Escolha uma pergunta ou escreva a sua.",
  "evidence_empty": "Não encontrei relações suficientes para esta pergunta.",
  "type_language": "Língua", "type_family": "Família ou subgrupo",
  "families_related": "Famílias e subgrupos", "more_info": "Para saber mais",
  "source": "Link {n}", "family_meta": "{n} línguas associadas",
  "card_language": "Neste projeto, esta língua ou variedade aparece {location}{family}. Você pode perguntar ao assistente para saber mais.",
  "card_family": "Este agrupamento reúne {n} línguas ou variedades registradas no grafo. Você pode explorar seus integrantes.",
  "card_location": "associada a {countries}", "card_no_location": "com um registro próprio", "card_family_suffix": " e relacionada com {families}",
  "family_label": "Família(s)",
  "kind": "Tipo", "languages": "Línguas", "families": "Famílias",
  "country": "País", "all_countries": "Todos", "family_filter": "Família linguística", "all_families": "Todas",
  "filter": "Buscar por nome ou ISO", "filter_placeholder": "Quechua, kariri, jaqaru…",
  "results": "{n} resultados · mostrando até 50", "download": "⬇️ Baixar seleção em CSV",
  "metric_languages": "Línguas e variedades", "metric_families": "Famílias e subgrupos", "metric_described": "Línguas com descrição",
  "about_project_title": "O projeto",
  "about_project": "Este projeto surgiu como uma forma de conversar com dados linguísticos especializados. Parte especialmente dos registros e classificações do Glottolog, organizados em um grafo de conhecimento, para fazer perguntas em linguagem natural e explorar relações entre línguas, agrupamentos e países. É mais uma maneira de consultar as informações disponíveis.",
  "about_method_title": "Como funciona",
  "about_method": "A aplicação utiliza um grafo de conhecimento para organizar informações sobre línguas e suas relações. Quando você faz uma pergunta, o sistema procura dados relevantes no grafo e utiliza um modelo de linguagem para elaborar uma resposta em espanhol ou português.",
  "about_sources_title": "Fontes de informação",
  "about_sources": "O grafo combina registros e classificações do Glottolog com informações do Wikidata e descrições da Wikipédia. Alguns textos também contêm referências ao Ethnologue; não se trata de uma importação independente dessa base de dados. Cada recurso contribui com informações diferentes.",
  "about_technical_title": "Arquitetura técnica",
  "about_technical": "Os dados são organizados em RDF (Resource Description Framework). A recuperação combina relações explícitas do grafo e busca textual com BM25, com opção de busca semântica por embeddings multilíngues. As respostas são geradas por um modelo de linguagem acessado pela API da Groq.",
  "about_scope_title": "Alcance",
  "about_scope": "O grafo não é um inventário completo das línguas indígenas da América do Sul. Algumas informações podem estar ausentes ou desatualizadas, e as estimativas do número de falantes devem ser interpretadas com cautela.",
  "chat_section_title": "Consultar o grafo",
  "chat_section_help": "Selecione uma pergunta ou escreva a sua.",
  "explore_intro": "Explore os registros passo a passo: escolha um país, um agrupamento e, se quiser, um subgrupo ou uma língua. Você também pode pesquisar diretamente por nome ou código ISO.",
  "explore_country_help": "Selecione um país para ver os agrupamentos e as línguas associados.",
  "explore_country_placeholder": "Selecione um país…", "explore_group": "Agrupamento principal",
  "explore_group_placeholder": "Todos os agrupamentos do país", "explore_subgroup": "Subgrupo (opcional)",
  "explore_subgroup_placeholder": "Todos os subgrupos", "explore_deeper": "Grupo mais específico (opcional)",
  "explore_language": "Língua ou variedade", "explore_language_placeholder": "Selecione uma língua…",
  "explore_search_help": "Pesquisar diretamente pelo nome ou código ISO",
  "explore_no_matches": "Não há línguas associadas a esta combinação.",
  "explore_count": "{n} línguas ou variedades nesta seleção.",
  "explore_select": "Escolha uma língua para ver sua ficha ou use a pesquisa.",
  "explore_no_country": "Escolha um país para começar ou escreva o nome de uma língua na pesquisa.",
  "explore_ungrouped": "Sem agrupamento registrado ({n})",
  "explore_groups_count": "{n} agrupamentos principais para este país.",
  "about_responsible_title": "Responsável e sugestões",
  "about_responsible": "Responsável: Javier Maximiliano Vera Zúñiga. Você pode escrever para compartilhar comentários ou sugestões sobre o projeto.",
  "source_links_label": "Recursos e links",
  "footer": "Línguas indígenas · América do Sul"
 }
}

COUNTRY_DISPLAY = {
 "pt": {"Perú": "Peru", "Colombia": "Colômbia", "Bolivia": "Bolívia", "Ecuador": "Equador", "Paraguay": "Paraguai", "Chile": "Chile", "Surinam": "Suriname", "Panamá": "Panamá", "Argentina": "Argentina", "Venezuela": "Venezuela", "Guyana": "Guiana", "Uruguay": "Uruguai", "Francia": "França", "Brasil": "Brasil", "Trinidad y Tobago": "Trinidad e Tobago"}
}

def t(language: str, key: str, **kwargs) -> str:
    # Una traducción ausente en un idioma utiliza la equivalente en español.
    # Los textos del explorador introducidos en v6.5 también cuentan con
    # respaldo en app_lenguas.py para actualizaciones parciales.
    template = STRINGS.get(language, {}).get(key)
    if template is None:
        template = STRINGS.get("es", {}).get(key)
    if template is None:
        raise KeyError(key)
    return template.format(**kwargs)

def country_label(language: str, country: str) -> str:
    return COUNTRY_DISPLAY.get(language, {}).get(country, country)
