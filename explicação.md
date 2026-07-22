# Relatorio Explicativo da Reproducao Parcial do Paper GLiNER

## 1. Objetivo deste documento

Este documento foi escrito para explicar, de forma clara e acessivel, o que este repositorio faz, por que ele existe e como os experimentos foram conduzidos.

A ideia central e simples:

- o paper original propoe um modelo chamado **GLiNER**
- esse modelo tenta resolver o problema de **Named Entity Recognition (NER)** de um jeito mais flexivel
- em vez de depender apenas de classificadores fechados ou de LLMs generativas muito grandes, ele usa um modelo menor e mais direto, mas ainda capaz de receber tipos de entidade como texto

Nosso trabalho aqui **nao foi reproduzir exatamente os numeros do paper**, mas sim **reproduzir corretamente a metodologia principal** usando a biblioteca publica do GLiNER e um pipeline de avaliacao rigoroso.

Em outras palavras:

- o foco foi a **fidelidade metodologica**
- nao a copia exata do treinamento original

---

## 2. O que e NER?

**NER** significa **Named Entity Recognition**, ou **Reconhecimento de Entidades Nomeadas**.

Em termos simples, NER e a tarefa de encontrar, dentro de um texto, trechos que representam entidades importantes, como:

- pessoas
- empresas
- lugares
- produtos
- organizacoes
- eventos

Por exemplo, na frase:

> "Maria trabalha na OpenAI em San Francisco."

um sistema de NER pode identificar:

- `Maria` como `pessoa`
- `OpenAI` como `organizacao`
- `San Francisco` como `local`

### Por que isso importa?

NER e util porque ajuda computadores a entender melhor o que aparece em textos.

Isso e importante em tarefas como:

- busca inteligente
- organizacao automatica de documentos
- analise de noticias
- sistemas juridicos e medicos
- assistentes virtuais
- mineracao de informacao
- criacao de bases de conhecimento

Sem NER, o computador ve apenas palavras. Com NER, ele comeca a entender **quem**, **onde**, **o que** e **sobre o que** o texto esta falando.

---

## 3. O problema classico do NER

Historicamente, muitos sistemas de NER funcionavam bem apenas quando:

- o conjunto de tipos era fixo
- o modelo era treinado especificamente para aqueles tipos
- o dominio do texto era parecido com o dominio do treino

Isso cria um limite importante.

Se um modelo foi treinado para reconhecer apenas:

- `PER`
- `ORG`
- `LOC`
- `MISC`

ele pode ter dificuldade quando queremos identificar categorias novas, como:

- `album`
- `cientista`
- `partido politico`
- `obra literaria`

Ou seja, o NER classico costuma ser bom em cenarios fechados, mas menos flexivel em cenarios abertos.

---

## 4. Por que o GLiNER foi um achado importante?

O GLiNER foi importante porque mostrou que e possivel fazer **NER mais flexivel** sem depender necessariamente de uma LLM generativa enorme.

### A intuicao central do GLiNER

Em vez de tratar NER como um problema com um conjunto fixo de classes, o GLiNER recebe os tipos de entidade como **texto em linguagem natural**.

Por exemplo, em vez de assumir que o modelo conhece apenas os codigos `PER`, `ORG` e `LOC`, podemos fornecer algo como:

- `person`
- `organization`
- `location`

ou ate descricoes mais livres, como:

- `human being`
- `company`
- `geographical location`

Isso aproxima o NER de um comportamento mais geral e mais reutilizavel.

### Por que isso e importante na pratica?

Porque o GLiNER mostrou que um modelo encoder-based, menor e mais direto, pode:

- aceitar tipos de entidade descritos em texto
- funcionar em modo zero-shot
- ser muito mais leve do que LLMs generativas
- ainda assim manter desempenho competitivo em varios benchmarks

### Como ele se torna uma boa alternativa a LLMs?

A grande contribuicao do GLiNER nao foi "vencer todas as LLMs em todos os cenarios".

A contribuicao real foi mostrar que:

- nem toda tarefa flexivel de extracao precisa de uma LLM autoregressiva gigante
- um modelo especializado pode ser muito mais barato e rapido
- e possivel obter um equilibrio muito interessante entre **custo**, **latencia**, **portabilidade** e **qualidade**

Em termos leigos:

- uma LLM costuma ser como um "canivete suico muito caro"
- o GLiNER e como uma "ferramenta especializada muito eficiente"

Se a tarefa principal e reconhecer entidades, muitas vezes faz mais sentido usar a ferramenta especializada.

---

## 5. O que os autores do paper propuseram

O paper propoe, em linhas gerais, o seguinte:

1. Treinar um modelo de NER geral em larga escala.
2. Representar os tipos de entidade usando texto natural.
3. Fazer predicao de spans no texto, em vez de depender apenas de rotulos token a token.
4. Avaliar por correspondencia exata entre:
   - fronteira da entidade
   - tipo da entidade

Esse ponto e muito importante:

o paper nao esta interessado apenas em "acertar mais ou menos a regiao". Ele exige que a entidade prevista tenha:

- o **span correto**
- e o **tipo correto**

Nosso repositorio foi ajustado justamente para ficar o mais perto possivel desse criterio.

---

## 6. O que exatamente nos replicamos

Nossa reproducao e **parcial**, mas foi desenhada para ser **metodologicamente fiel**.

### O que replicamos de forma mais proxima

- uso de labels em linguagem natural
- inferencia baseada em spans com a API publica do GLiNER
- avaliacao principal por exact match de span + tipo
- experimentos zero-shot e cross-domain
- comparacao no estilo da tabela OOD do paper

### O que nao foi replicado integralmente

- o treinamento original do paper
- o corpus Pile-NER usado pelos autores
- as heuristicas internas de treino descritas no artigo
- a implementacao interna exata do modelo dos autores

Em vez disso, usamos **checkpoints publicos ja treinados** da biblioteca GLiNER.

Isso significa que nosso objetivo foi:

- reproduzir corretamente o **pipeline experimental**
- nao reconstruir todo o **processo de pre-treinamento**

---

## 7. Melhorias metodologicas feitas no repositorio

Antes desta revisao, o repositorio tinha uma base funcional, mas faltavam alguns pontos importantes para uma reproducao mais confiavel.

As principais melhorias foram:

### 7.1 Reconstrucao de texto com offsets

Muitos datasets no ecossistema `datasets` fornecem:

- tokens
- tags BIO

mas nao fornecem o texto cru exatamente como o modelo o receberia.

Como o GLiNER devolve spans por caractere, foi necessario reconstruir o texto de maneira deterministica e manter offsets coerentes.

Isso foi implementado para aproximar melhor a avaliacao do comportamento real do modelo.

### 7.2 Avaliacao por exact match de span + tipo

A avaliacao principal passou a seguir a logica mais alinhada ao paper:

- uma entidade so conta como correta se o span estiver certo
- e se o tipo tambem estiver certo

Essa foi provavelmente a correcao metodologica mais importante de toda a reproducao.

### 7.3 Separacao entre label de prompt e label canonico

O modelo pode receber labels mais "humanos" como:

- `human`
- `company`
- `country`

mas a avaliacao precisa continuar coerente com o schema do dataset.

Por isso, o pipeline faz o mapeamento entre:

- o texto do prompt
- e o rotulo canonico de avaliacao

### 7.4 Pipeline reutilizavel

Os experimentos passaram a compartilhar:

- carregamento de dataset
- reconstrucao de texto
- inferencia
- avaliacao
- logging
- escrita de CSV

Isso torna a replicacao mais clara, modular e auditavel.

---

## 8. Metricas utilizadas

Para uma pessoa leiga, vale entender essas metricas assim:

### Precision

Das entidades que o modelo disse que existiam, quantas estavam certas?

Se a precision e baixa, o modelo "alucina" demais.

### Recall

Das entidades que realmente existiam no texto, quantas o modelo conseguiu encontrar?

Se o recall e baixo, o modelo deixa passar muita coisa.

### F1

E a media harmonica entre precision e recall.

Ela e importante porque obriga um equilibrio:

- nao adianta prever tudo e errar muito
- tambem nao adianta prever pouco demais e perder entidades reais

### Exact matches

Numero de entidades acertadas completamente:

- inicio correto
- fim correto
- tipo correto

### Predicted entities

Quantas entidades validas o modelo previu ao final da avaliacao.

### Reference entities

Quantas entidades existiam de fato no gabarito.

### Raw predicted entities

Quantas entidades o modelo produziu antes do filtro final de avaliacao.

### Time total e time average

- `time_seconds_total`: tempo total do experimento
- `time_seconds_average`: tempo medio por exemplo

### Classification report

O `classification_report` foi mantido como relatorio auxiliar por classe.

Mas e importante reforcar:

**a metrica principal da replicacao e a avaliacao por exact match de span + tipo**, e nao apenas o relatorio BIO.

---

## 9. Maquina utilizada

Os experimentos locais desta replicacao foram executados em:

- CPU: `Intel Core i7-1355U`
- Memoria RAM: `15 GiB`
- GPU NVIDIA: nao disponivel no ambiente

Isso importa porque:

- os tempos observados aqui refletem execucao em CPU
- portanto, eles servem bem para comparacao interna entre experimentos
- mas nao devem ser lidos como benchmark universal de velocidade

---

## 10. Datasets usados ao longo da replicacao

Foram usados **9 datasets/unidades de benchmark**:

1. `CoNLL-2003`
2. `MIT Movie`
3. `MIT Restaurant`
4. `CrossNER-AI`
5. `CrossNER-Literature`
6. `CrossNER-Music`
7. `CrossNER-Politics`
8. `CrossNER-Science`
9. `TweetNER7 (2021)`

Esses datasets foram usados para cobrir tres objetivos:

- validar o pipeline
- estudar sensibilidade a prompts
- aproximar os cenarios zero-shot e OOD discutidos no paper

---

## 11. Modelos considerados na replicacao

### Modelos executados localmente

- `GLiNER Large`
- `GLiNER Base`
- `GLiNER Small`

### Modelos usados como referencia comparativa do paper

- `UniNER-7B`
- `GoLLIE`

Esses dois ultimos foram usados como comparacao de tabela no experimento OOD, com numeros reportados no paper, e nao rerodados localmente nesta maquina.

Isso foi uma escolha consciente para manter honestidade cientifica:

- executamos localmente o GLiNER
- comparamos com baselines open-source do paper
- mas sem fingir que rerodamos todos os grandes modelos instruction-tuned

---

## 12. Experimento 1 - Zero-shot principal em CoNLL-2003

### O que esse experimento e?

Este e o experimento mais simples da replicacao.

A ideia foi verificar se o pipeline basico estava funcionando:

- carregar o dataset
- reconstruir o texto
- enviar labels textuais ao GLiNER
- recuperar spans
- converter spans para a estrutura de avaliacao
- medir exact match

### Por que ele e necessario?

Porque sem esse teste inicial nao seria seguro avancar para benchmarks maiores.

Ele funciona como um "teste de sanidade metodologica".

### Como foi medido?

- dataset: `CoNLL-2003`
- split: `test`
- labels: `person`, `organization`, `location`, `miscellaneous`
- checkpoint: `urchade/gliner_large-v2.1`
- exemplos avaliados: `10` | `3453`
- threshold: `0.5`

### Resultado

- Precision: `0.5000` |  `0.4730`
- Recall: `0.5714` | `0.6510`
- F1: `0.5333` | `0.5479`
- Exact matches: `12`
- Entidades previstas: `24`
- Entidades reais: `21`

### Debate dos resultados

O resultado foi suficiente para mostrar que o pipeline estava funcional.

No entanto, esse experimento e pequeno demais para sustentar conclusoes fortes sobre desempenho.

O que ele nos disse foi:

- o modelo consegue encontrar entidades em zero-shot
- o mapeamento de offsets e labels estava coerente
- o criterio de exact match estava sendo aplicado corretamente

Em resumo:

esse experimento e importante mais como **validacao de infraestrutura** do que como comparacao cientifica final.

---

## 13. Experimento 2 - Label Prompt Experiment

### O que esse experimento e?

Aqui queriamos descobrir se a forma de descrever os tipos de entidade altera o desempenho.

O GLiNER recebe labels em texto. Portanto, trocar:

- `person`

por

- `human`

ou

- `PER`

pode influenciar o resultado.

### Por que ele e necessario?

Porque essa e uma parte central da proposta do GLiNER.

Se o modelo trabalha com tipos descritos em linguagem natural, entao o texto do label deixa de ser apenas um detalhe e passa a ser parte da metodologia.

### Configuracoes testadas

1. `PER ORG LOC MISC`
2. `human being organization geographical location miscellaneous named entity`
3. `person organization location miscellaneous`
4. `human company country other entity`

### Como foi medido?

- dataset: `CoNLL-2003`
- split: `test`
- checkpoint: `urchade/gliner_large-v2.1`
- exemplos avaliados: `10` | `400`
- threshold: `0.5`

### Resultados

- `PER/ORG/LOC/MISC`: `F1 = 0.0714` | `0.1789`
- `descriptive phrases`: `F1 = 0.2500` | `0.6249`
- `natural labels`: `F1 = 0.5333` | `0.6620`
- `semantic aliases`: `F1 = 0.6667` | `0.5722`

### Debate dos resultados

Este foi um dos experimentos mais reveladores.

Ele mostrou que:

- o wording do prompt importa muito
- labels "naturais" ou semanticamente proximos podem funcionar melhor do que siglas
- siglas como `PER`, `ORG`, `LOC` nao sao automaticamente a melhor escolha para o GLiNER

Esse resultado ajuda a explicar por que reproducoes desse tipo de paper podem variar tanto:

- pequenas mudancas na descricao das classes podem alterar bastante o F1

Ou seja, esse experimento reforca uma mensagem importante:

**em GLiNER, o prompt de labels faz parte da engenharia metodologica, e nao apenas da interface.**

---

## 14. Experimento 3 - Compare Models

### O que esse experimento e?

Esse experimento compara tamanhos diferentes do GLiNER:

- Large
- Base
- Small

### Por que ele e necessario?

Porque nem sempre o maior modelo e automaticamente o melhor em um conjunto muito pequeno ou em um prompt especifico.

Tambem queriamos observar o equilibrio entre:

- qualidade
- custo computacional
- velocidade

### Como foi medido?

- dataset: `CoNLL-2003`
- split: `test`
- labels: `person`, `organization`, `location`, `miscellaneous`
- exemplos avaliados: `10` | `500`
- threshold: `0.5`

### Resultados

- `GLiNER Large`: `F1 = 0.5333` | `0.6939`, tempo medio `0.3391 s` | `0.1357 s`
- `GLiNER Base`: `F1 = 0.6087` | `0.7384`, tempo medio `0.1062 s` | `0.0473 s`
- `GLiNER Small`: `F1 = 0.5641` | `0.7338`, tempo medio `0.0600 s` | `0.0299 s`

### Debate dos resultados

O resultado mais curioso foi o `GLiNER Base` ter ficado acima do `Large`.

Isso nao significa que o modelo Base seja melhor em geral.

Significa apenas que, neste recorte:

- o Base teve o melhor equilibrio

O valor real desse experimento esta em mostrar que:

- ha trade-off entre tamanho e velocidade
- checkpoints menores podem ser muito interessantes em cenarios praticos

---

## 15. Experimento 4 - Benchmark OOD no estilo da Tabela 1

### O que esse experimento e?

Este foi o experimento mais importante da replicacao.

OOD significa **Out-of-Domain**.

Em palavras simples:

- treinamos ou usamos o modelo de forma geral
- e depois testamos em dominios diferentes

Aqui os dominios avaliados foram:

- Movie
- Restaurant
- AI
- Literature
- Music
- Politics
- Science

### Por que ele e necessario?

Porque esse tipo de avaliacao testa justamente a proposta central do GLiNER:

- generalizar para tipos e dominios diferentes
- sem precisar de um modelo fechado para apenas um schema fixo

### Como foi medido?

- checkpoint: `urchade/gliner_large-v2.1`
- threshold: `0.5`
- `200` exemplos por dominio
- total de `1400` exemplos avaliados

Tambem registramos:

- tempo total
- tempo medio por exemplo
- numero de entidades previstas
- numero de entidades reais

### Resultados locais do GLiNER na reproducao

- `Movie`: `44.6`
- `Restaurant`: `46.6`
- `AI`: `52.8`
- `Literature`: `57.8`
- `Music`: `73.4`
- `Politics`: `64.0`
- `Science`: `60.4`
- `Average`: `57.1`

### Comparacao com resultados reportados no paper

- `GLiNER-L (paper)`: `60.9`
- `UniNER-7B`: `53.7`
- `GoLLIE`: `58.0`

### Debate dos resultados

Este experimento foi a melhor evidencia de que a reproducao ficou **proxima da narrativa do paper**, ainda que nao identica.

Pontos positivos:

- nossa media `57.1` ficou perto da faixa dos modelos fortes da tabela
- o desempenho ficou acima do `UniNER-7B` reportado
- o resultado ficou muito proximo de `GoLLIE`
- em `Music`, `Restaurant` e `Science`, o comportamento foi especialmente competitivo

Pontos de atencao:

- ficamos abaixo da linha `GLiNER-L` do paper
- `Movie` foi um dominio claramente dificil
- `Politics` e `Literature` tambem ficaram abaixo da referencia do artigo

Esse resultado faz sentido por varios motivos:

1. Nao usamos exatamente o mesmo modelo congelado do paper.
2. Nao reproduzimos o treino original em Pile-NER.
3. Usamos a biblioteca publica, e nao a implementacao de pesquisa completa dos autores.
4. Avaliamos `200` exemplos por dominio, e nao necessariamente todo o conjunto do artigo.
5. O wording dos labels e a reconstrucao de texto tambem influenciam.

Mesmo assim, este foi um resultado muito encorajador:

- ele sugere que a reproducao capturou a essencia experimental do paper
- sem mascarar as diferencas inevitaveis

---

## 16. Experimento 5 - Zero-shot adicional em TweetNER7

### O que esse experimento e?

Este experimento testa o GLiNER em um conjunto de tweets.

Tweets sao mais dificeis porque costumam ter:

- texto curto
- linguagem informal
- abreviacoes
- ruído
- referencias contextuais

### Por que ele e necessario?

Porque ele complementa o benchmark OOD com um tipo de texto mais "real" e mais desordenado.

Se o modelo vai ser realmente geral, ele precisa lidar tambem com esse cenario.

### Como foi medido?

- dataset: `TweetNER7 (2021)`
- checkpoint: `urchade/gliner_large-v2.1`
- exemplos avaliados: `300`
- threshold: `0.5`

### Resultado

- Precision: `0.3776`
- Recall: `0.4616`
- F1: `0.4154`
- Entidades previstas: `1099`
- Entidades reais: `899`
- Exact matches: `415`

### Debate dos resultados

Esse foi um cenario claramente mais dificil do que varios dominios do CrossNER.

O modelo teve:

- recall razoavel
- precision mais baixa

Isso sugere uma tendencia a:

- encontrar bastante coisa
- mas tambem prever entidades demais em texto ruidoso

Em termos simples:

o modelo ficou mais "agressivo" no Twitter.

Isso nao invalida o GLiNER.

Na verdade, mostra algo importante:

- a proposta do modelo e forte
- mas o nivel de dificuldade muda bastante de acordo com o dominio

Esse experimento reforca um ponto central de reproducao:

**generalizacao aberta nao significa desempenho uniforme em qualquer tipo de texto.**

---

## 17. O que aprendemos ao comparar todos os experimentos

Ao olhar os experimentos em conjunto, algumas licoes ficaram muito claras.

### 17.1 O pipeline de avaliacao importa muito

Se a avaliacao nao respeita:

- spans
- offsets
- mapeamento de labels
- exact match de tipo + fronteira

entao a reproducao pode gerar um numero "bonito", mas cientificamente fraco.

### 17.2 Prompt de labels nao e detalhe

O experimento de label prompt mostrou que a descricao textual das entidades muda o resultado de forma drastica.

Isso tem implicacao direta para zero-shot NER.

### 17.3 GLiNER e forte sem ser uma LLM gigante

O benchmark OOD mostrou que o GLiNER continua muito competitivo, inclusive quando comparado com baselines open-source reportados no paper.

Essa e uma das mensagens mais importantes da reproducao.

### 17.4 Dominio faz diferenca

O desempenho muda bastante entre:

- noticias
- filmes
- politica
- musica
- tweets

Portanto, qualquer conclusao sobre "qualidade do modelo" precisa sempre respeitar o tipo de dado avaliado.

---

## 18. Por que os resultados ficaram parecidos com o paper, mas nao iguais?

Essa e talvez a pergunta mais importante em qualquer reproducao cientifica.

### Por que houve semelhanca?

Houve semelhanca porque conseguimos reproduzir os elementos mais importantes da metodologia:

- labels textuais
- zero-shot
- avaliacao por exact match
- benchmark cross-domain
- comparacao OOD

Ou seja, capturamos a **estrutura experimental correta**.

### Por que nao ficou igual?

Porque reproduzir um paper nunca significa apenas rodar um script.

As diferencas principais foram:

1. **Modelo diferente**
   - usamos checkpoints publicos atuais
   - nao exatamente o mesmo artefato do paper

2. **Treinamento nao reproduzido**
   - nao repetimos o pre-treinamento em Pile-NER

3. **Biblioteca publica**
   - usamos a API publica do GLiNER
   - nao a pilha completa de pesquisa dos autores

4. **Reconstrucao de texto**
   - parte dos datasets precisou ser reconstruida a partir de tokens

5. **Amostragem parcial**
   - alguns experimentos foram exploratorios
   - especialmente `CoNLL` com `10` exemplos

6. **Sensibilidade ao prompt**
   - pequenas mudancas nos labels alteram o F1

Essas diferencas nao sao falhas escondidas.

Elas sao justamente o tipo de limitacao que uma reproducao honesta deve declarar.

---

## 19. Conclusao final da reproducao

Esta reproducao parcial foi bem-sucedida no que se propunha a fazer.

Ela nao recriou integralmente o treino do paper, mas conseguiu reproduzir com boa fidelidade:

- a ideia central do GLiNER
- o uso de labels em linguagem natural
- o pipeline zero-shot
- a avaliacao por exact match
- o benchmark OOD no estilo da Tabela 1

### O que podemos concluir sobre o GLiNER?

Podemos concluir que o GLiNER:

- e uma proposta metodologicamente muito relevante
- oferece uma alternativa realista a LLMs generativas em NER
- combina flexibilidade com custo computacional mais razoavel
- continua competitivo em varios dominios mesmo em uma reproducao parcial

### O que podemos concluir sobre esta replicacao?

Podemos concluir que esta replicacao:

- ficou cientificamente mais rigorosa do que uma simples execucao de exemplo
- documentou melhor as aproximacoes inevitaveis
- produziu resultados coerentes com a narrativa geral do paper
- mostrou claramente onde houve semelhanca e onde houve divergencia

### Mensagem final em linguagem simples

Se uma pessoa totalmente de fora da area perguntasse:

> "Entao, afinal, o que voces provaram com essa reproducao?"

a resposta seria:

> "Mostramos que a principal ideia do paper funciona tambem em uma reproducao parcial e transparente: um modelo especializado, menor e mais eficiente que uma LLM generativa pode fazer NER de forma flexivel usando labels em linguagem natural, desde que o pipeline experimental seja montado com cuidado."

---

## 20. Resumo executivo

Para fechar, aqui vai a versao curta de tudo:

- NER e a tarefa de encontrar entidades importantes em um texto.
- O GLiNER foi importante porque mostrou que da para fazer NER flexivel sem depender apenas de LLMs gigantes.
- Nossa reproducao focou em metodologia, nao em copiar exatamente o treino dos autores.
- O repositorio foi ajustado para avaliar por exact match de span + tipo.
- O wording dos labels influenciou muito o desempenho.
- O benchmark OOD foi o experimento mais importante e mostrou resultado local medio de `57.1`, proximo da faixa forte do paper.
- O TweetNER7 confirmou que texto ruidoso continua sendo um desafio.
- No geral, a reproducao sustenta a principal intuicao do artigo e mostra que o GLiNER e uma alternativa pratica, eficiente e cientificamente interessante para NER zero-shot.
