# Туториалы
* [Раздел туториалов на PyTorch](https://pytorch.org/tutorials/index.html) - все материалы подаются через призму Torch: как сети составляются, как обучаются, объясняются необходимые примитивы типа Dataset и DataLoader и т.д. Есть разделы по модальностями. Правда, они завязывают работу с модальностями на их библиотеки типа torchtext и torchvision.
* [Раздел туториалов на Transformers](https://huggingface.co/learn/nlp-course/chapter1/1) - кратко расскажут, что такое NLP и как в ней применяются трансформеры. Объясняется, как работать с библиотекой transformers: основные примитивы, как использовать модели, как их дообучать, показывают, как использовать это все для разных задач NLP.
* [Блог Джея Аламара с визуальными туториалмами](http://jalammar.github.io) — в блоге содержится много важных концептов с богатой визуализацией, которая сильно помогает понять, что вообще происходит. Список постов, относящихся к NLP:
  * [The Illustrated Retrieval Transformer](http://jalammar.github.io/illustrated-retrieval-transformer/) ([rus](https://habr.com/ru/post/648705/))
  * [How GPT3 Works - Visualizations and Animations](http://jalammar.github.io/how-gpt3-works-visualizations-animations/) ([rus](https://habr.com/ru/articles/514698/))
  * [A Visual Guide to Using BERT for the First Time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) ([rus](https://habr.com/ru/articles/498144/))
  * [The Illustrated GPT-2 Visualizing Transformer Language Models](http://jalammar.github.io/illustrated-gpt2/) ([rus](https://habr.com/ru/articles/490842/))
  * [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/) ([rus](https://habr.com/ru/articles/446530/))
  * [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/) ([rus](https://habr.com/ru/articles/487358/))
  * [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) ([rus](https://habr.com/ru/articles/486358/))
  * [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) ([rus](https://habr.com/ru/articles/486158/))
  * [A Visual And Interactive Look at Basic Neural Network Math](http://jalammar.github.io/feedforward-neural-networks-visual-interactive/)
  * [A Visual and Interactive Guide to the Basics of Neural Networks](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)
* [Transformers Inference Optimization Toolset](https://astralord.github.io/posts/transformer-inference-optimization-toolset) — пост написан для того, чтобы дать основы по оптимизации инференса, чтобы быть готовым погружаться в глубины тематики.
* [Linkage Clustering](https://github.com/mustafa-sadeghi/Linkage-Clustering) — this repository contains a comprehensive guide on Linkage Clustering, a technique used in hierarchical clustering within unsupervised machine learning. It combines theoretical insights with practical Python implementations for easy learning.
* [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — When looking at the latest releases of Large Language Models (LLMs), you will often see “MoE” in the title. What does this “MoE” represent and why are so many LLMs using it? In this visual guide, we will take our time to explore this important component, Mixture of Experts (MoE) through more than 50 visualizations!

# Книги
* [Natural Language Processing with Transformers, Revised Edition, 2022](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - книга про трансформеры. Разбирается устройство а также его применение на типовые задачи типа классификации, NER, генерации текста, саммаризации, QA и еще несколько прикладных глав. Подробное содержание по ссылку.
* [Учебник по машинному обучению](https://academy.yandex.ru/handbook/ml) - основательный учебник от Яндексовской Школы Анализа Данных, в котором разбираются как классические алгоритмы машинного обучения, так и нейросетевые архитектуры. По содержанию кажется, что упоров на какую-либо модальность нет.
* [Deep Learning](https://www.deeplearningbook.org/) - учебник от MIT Press по основам машинного обучения. Включает три главы: в первой дается необходимая математика, во второй разбираются базовые архитектуры нейросетей, хотя и без трансформеров и графовых сетей, в третьей обозреваются разные проблемы и направления глубокого обучения.
* [Глубокое обучение. Погружение в мир нейронных сетей](https://www.litres.ru/book/a-kadurin-13464223/glubokoe-obuchenie-pogruzhenie-v-mir-neyronnyh-setey-29817855/) - в книге разбираются подробно разные архитектуры, которые были на 2018 год. Имеются примеры кода на TensorFlow и Keras, только устаревшие.
* [Natural Language Processing with Python](https://www.nltk.org/book/) - книга, идущая в довесок к библиотеке NLTK, показывает основные методики обработки текстов без сложных моделей машинного обучения. Первые главы посвещаны особенностям программирования на Python, затем обсуждаются частеречная разметка, классификацию текстов, синтаксический анализ и другое.
* [\[Хабр\] Пять книг про NLP, с которых можно начать](https://habr.com/ru/companies/ru_mts/articles/759266/) - подборка от Валентина Малых, известного NLP-специалиста, который много лет проводит курсы по NLP.
* [\[Хабр\] Machine Learning: хорошая подборка книг для начинающего специалиста](https://habr.com/ru/companies/ru_mts/articles/759266/) - подборка книг по общему машинному обучению с использованием Python.
* [Introduction to Transformers: an NLP Perspective](https://arxiv.org/pdf/2311.17633.pdf) — небольшая книга (я бы назвал методичка), в которой подробно описана первоначальная архитектура Трансформера, различные улучшения к ней, накопленные с годами, а также задачи, где применяется эта архитектура.
* [\[Хабр\] Прокачиваем навыки в сфере ML — что изучать в 2024-м](https://habr.com/ru/companies/cloud_mts/articles/787386/) — книжки по NLP и общей машинке.
* [Математическая статистика](https://github.com/astralord/Statistics-lectures/blob/master/book.pdf) - перевод на русский язык [серии постов](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/). Оригинальная аннотация: "This series of posts is a guidance for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics."

# Курсы
* [karpov.courses](https://karpov.courses) — платформа курсов. Есть [бесплатный курс](https://karpov.courses/mathsds) по математике для DS.
* [Лучшие бесплатные курсы и ресурсы для изучения машинного обучения](https://habr.com/ru/articles/804251/) - в этой статье автор собрал огромную коллекцию курсов, книг, и ресурсов для всех, кто любит и изучает машинное обучение.
* [Generative AI for beginners](https://github.com/microsoft/generative-ai-for-beginners) — Learn the fundamentals of building Generative AI applications with our 18-lesson comprehensive course by Microsoft Cloud Advocates.
* [smol-course](https://github.com/huggingface/smol-course) - бесплатный открытый курс от Hugging Face по файнтюнингу больших языковых моделей. В курсе рассматриваются теория и практические аспекты работы с такими методами, как LoRA, супервайзед-файнтюнинг, DPO, ORPO и другие техники для настройки моделей под конкретные задачи.
* [Курсы от Alexander Avdiushenko](https://avalur.github.io/teaching.html) — набор курсов по основам машинного обучения и Python.
* [Deep Learning course, University at Buffalo](https://cedar.buffalo.edu/~srihari/CSE676/) —  The course, which will be taught through lectures and projects, will cover the underlying theory, the range of applications to which it has been applied, and learning from very large data sets. The course will cover connectionist architectures commonly associated with deep learning, e.g., basic neural networks, convolutional neural networks and recurrent neural networks. Methods to train and optimize the architectures and methods to perform effective inference with them, will be the main focus.
* [Современный NLP. Большие языковые модели](https://github.com/dmkalash/mailru_llm_course) — открытые материалы курса от Mail.ru на Github.

## Видеокурсы
* [Stanford XCS224U: Natural Language Understanding](https://www.youtube.com/watch?v=K_Dh0Sxujuc&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp) ([материалы на GitHub](https://github.com/cgpotts/cs224u/)) - покрывает все основные темы по пониманию языка (крупная подобласть NLP). В основном, видео идут до 20 минут.
* [Введение в обработку естественного языка](https://stepik.org/course/1233/promo) - курс познакомит с разными задачами в обработке естественного языка. Именно познакомит, а не покажет как они решаются современными подходами. 
* [Нейронные сети и обработка текста](https://stepik.org/course/54098/promo) - курс о нейросетях в обработке естественного языка. Покрывает все базовые вопросы, но не рассказывает про БЯМ.
* Байесовские методы машинного обучения ([страница курса](http://www.machinelearning.ru/wiki/index.php?title=%D0%91%D0%BC%D0%BC%D0%BE), [youtube плейлист](https://www.youtube.com/playlist?list=PLEqoHzpnmTfCiJpMPccTWXD9DB4ERQkyw)) - модели машинного обучения через линзу байесовской статистики, EM-алгоритм, Байесовский вывод, метод Монте-Карло по схеме марковских цепей, LDA и т.д.
* [Основы статистики](https://stepik.org/course/76/info) — В рамках трехнедельного курса рассматриваются подходы к описанию получаемых в исследованиях данных, основные методы и принципы статистического анализа, интерпретация и визуализация получаемых результатов.
* [Основы комбинаторики и теории вероятностей для чайников](https://stepik.org/course/172817/promo) — Курс по основам комбинаторики для учащихся 9-11 классов, в котором вы познакомитесь с основными понятиями в комбинаторике и теории вероятности и научитесь решать задачи легкого и среднего уровня, а также сможете выбрать удобный для вас вид подачи материала (текст и видео).
* [Теория вероятностей](https://stepik.org/course/3089/info) — Курс знакомит слушателей с базовыми понятиями теории вероятностей: вероятностным пространством, условной вероятностью, случайными величинами, независимостью, математическим ожиданием и дисперсией. Доказываются закон больших чисел и некоторые версии предельных теорем. Разобрано много примеров и задач.
* [Deep Learning School](https://www.youtube.com/@DeepLearningSchool/featured) — Официальный канал школы "Deep Learning School" от Физтех-Школы прикладной математики и информатики МФТИ и Лаборатории нейронных систем и глубокого обучения МФТИ.
* [Samsung Innovation Campus Russia](https://www.youtube.com/@samsunginnovationcampusrussia/playlists) — куча всяких видеолекций по Data Science, в том числе по NLP и CV (см. плейлисты).
* [Курс «Машинное обучение» 2019 ШАД](https://www.youtube.com/playlist?list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK) — читает К. В. Воронцов.
* [Курс «Машинное обучение 1» (Евгений Соколов)](https://www.youtube.com/playlist?list=PLEqoHzpnmTfChItexxg2ZfxCsm-8QPsdS)


## Текстовые курсы
* Открытый курс машинного обучения ([рус](https://habr.com/ru/companies/ods/articles/322626/), [анг](https://mlcourse.ai/book/index.html)) - курс от OpenDataScience, в котором обозреваются общие алгоритмы машинного обучения.
* [Курс "Машинное обучение" на ФКН ВШЭ](https://github.com/esokolov/ml-course-hse) - конспекты лекций, материалы семинаров и домашние задания (теоретические, практические, соревнования) по курсу "Машинное обучение". Тщательно разбираются классические алгоритмы машинного обучения.
* [Курс про Трансформеры \[англ\]](https://github.com/s-nlp/transformers-course) — курс про архитектуру Трансформеров, кто имеет опыт в Питоне и в Глубоком обучении.

# Обзоры
* [О методах позиционного кодирования в Transformer](https://habr.com/ru/articles/780116/) - статья посвящёна проблеме выбора метода позиционного кодирования в нейросетевых моделях на основе архитектуры Transformer.
* [«ИИ без границ»: как научить Transformer обрабатывать длинные тексты](https://habr.com/ru/articles/773312/) - статья посвящёна проблеме обработки длинных входных последовательностей нейросетевыми моделями на основе архитектуры Transformer.
* [Зоопарк трансформеров: большой обзор моделей от BERT до Alpaca](https://habr.com/ru/companies/just_ai/articles/733110/) - статья содержит обзор моделей на бзе Transformer, вышедших в период с 2019 по 2023 год.

# Разное
* [Детальная картинка устройства энкодера Трансформера](https://github.com/pa-shk/transformer-encoder/blob/main/transformer-encoder.png)
* [Сообщество Open Data Science](https://ods.ai/) — соревнования, курсы, вакансии от самого крупного сообщества DS в России.
* [Список вопрос для интервью NLP](https://dynamic-epoch-4bb.notion.site/100-questions-about-NLP-549ccde0d81a4689b5635888b9d0d7e6) — сто вопросов, собранных авторами разных телеграм-каналов, которые можно ожидать на собеседовании по NLP.
* [Вопросы и ответы с собеседований](https://t.me/ai_machinelearning_big_data/4532) - большая, подборка вопросов и ответов с собеседований по ML, Data Science,Ai, статистике, теории вероятностей python, SQL.
* LLM Visualization ([сайт](https://bbycroft.net/llm), [код](https://github.com/bbycroft/llm-viz)) — интерактивная визуализация работы БЯМ с пояснениями.
* Теория вероятностей в машинном обучении ([часть 1](https://habr.com/ru/companies/ods/articles/713920/), [часть 2](https://habr.com/ru/companies/ods/articles/714670/)) — в первой части подробно рассматривается вероятностная постановка задачи машинного обучения. Во второй части рассматривается метод максимизации правдоподобия в классификации.
* [Промпт-инжиниринг: как найти общий язык с ИИ](https://habr.com/ru/companies/mts_ai/articles/844624/) — популярное введение в промт-инжиниринг.
* [Перплексия в языковых моделях](https://habr.com/ru/companies/wunderfund/articles/580230/) — в этом материале я хочу сделать подробный обзор такого понятия, как «перплексия». Я расскажу о двух подходах, которые обычно используются для определения этого понятия, и о тех идеях, которые лежат в основе этих подходов.
* [Как полюбить математику и подружиться с ней на всю жизнь, если ты уже не школьник](https://habr.com/ru/articles/896816/) — статья о том, как автор учил математику самостоятельно. Дано множество советов, а также материалов для изучения.
* [The probability cheatsheet](https://github.com/wzchen/probability_cheatsheet) — This cheatsheet is a 10-page reference in probability that covers a semester's worth of introductory probability. The cheatsheet is based off of Harvard's introductory probability course, Stat 110. It is co-authored by former Stat 110 Teaching Fellow William Chen and Stat 110 Professor Joe Blitzstein.

# Дорожные карты
* [Вкатываемся в Machine Learning с нуля за ноль рублей: что, где, в какой последовательности изучить](https://habr.com/ru/articles/774844/) — подробный гайд на тему того, как можно изучать Machine Learning самостоятельно, не тратя деньги на платные курсы.
* [Математика для взрослых. Дорожная карта от выпускника Хармфульского клуба математики](https://habr.com/ru/companies/gaz-is/articles/779998/) — краткий обзор программы для самостоятельного изучения математики (от школьной до университета).
* [50 исследований на тему нейросетей, которые помогут вам стать ИИ-инженером от бога](https://habr.com/ru/companies/magnus-tech/articles/867762/) — В этом дайджесте мы собрали 50 знаковых научных работ в области ИИ за последние годы (от января 2025 года). Подборка охватывает десять ключевых направлений разработки нейросетей: от промтинга и проектирования бенчмарков до файнтюнинга и компьютерного зрения. Материал будет полезен как для опытных ИИ-инженеров, которые хотят прокачать свои навыки разработки, так и тем, кто только начинает свое знакомство с нейросетями и находится в поисках точки входа в ту или иную тему.

# Хорошее про програмиирование

[Паттерны и практики написания кода](https://avito.tech/patterns) — бесплатный курс для программистов от бэкенд-инженера Авито — Юры Афанасьева. Курс посвящен практикам и паттернам написания кода. Он будет полезен как начинающим, так и middle-разработчикам.
	* [Курс «Паттерны и практики написания кода». Сезон 1](https://www.youtube.com/playlist?list=PLknJ4Vr6efQHD8qkPPosGQjqrZpTa7KQP) — Это первый сезон курса посвященного практикам и паттернам написания кода. Он будет полезен как начинающим, так и middle-разработчикам. Эти видеоролики являются частью большого курса, созданного специально для студентов МАИ и успешно проведены в учебном заведении.
	* [Курс «Паттерны и практики написания кода». Сезон 2](https://www.youtube.com/playlist?list=PLknJ4Vr6efQHvhvlGcBSD4KHa4ekAn0DS) — Это второй сезон курса о паттернах и практиках написания кода от бэкенд-инженера Авито — Юрия Афанасьева. Сезон состоит из 10 серий, которые будут выходить по вторникам. В каждой из них мы подробнее углубимся в принципы программирования и разберём их на примерах.
	* [Курс «Паттерны и практики написания кода»](https://www.youtube.com/playlist?list=PLknJ4Vr6efQEru1-My3BsnYw-4UZzA_-d) — Бесплатный курс для программистов от бэкенд-инженера Авито — Юры Афанасьева. Курс посвящен практикам и паттернам написания кода. Он будет полезен как начинающим, так и middle-разработчикам.

# Источники

Здесь буду хранить ссылки на подборки, из которых достал только определенные ссылки.
1. https://habr.com/ru/companies/jetinfosystems/articles/777632/
