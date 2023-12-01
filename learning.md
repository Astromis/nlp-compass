# Туториалы
* [Раздел туториалов на PyTorch](https://pytorch.org/tutorials/index.html) - все материалы подаются через призму Torch: как сети составляются, как обучаются, объясняются необходимые примитивы типа Dataset и DataLoader и т.д. Есть разделы по модальностями. Правда, они завязывают работу с модальностями на их библиотеки типа torchtext и torchvision.
* [Раздел туториалов на Transformers](https://huggingface.co/learn/nlp-course/chapter1/1) - кратко расскажут, что такое NLP и как в ней применяются трансформеры. Объясняется, как работать с библиотекой transformers: основные примитивы, как использовать модели, как их дообучать, показывают, как использовать это все для разных задач NLP.
* [Блог Джея Аламара с визуальными туториалмами](http://jalammar.github.io) — в блоге содержится много важных концептов с богатой визуализацией, которая сильно помогает понять, что вообще происходит. Список постов, относящихся к NLP:
  * [The Illustrated Stable Diffusion](http://jalammar.github.io/illustrated-stable-diffusion/)
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

# Книги
* [Natural Language Processing with Transformers, Revised Edition, 2022](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - книга про трансформеры. Разбирается устройство а также его применение на типовые задачи типа классификации, NER, генерации текста, саммаризации, QA и еще несколько прикладных глав. Подробное содержание по ссылку.
* [Учебник по машинному обучению](https://academy.yandex.ru/handbook/ml) - основательный учебник от Яндексовской Школы Анализа Данных, в котором разбираются как классические алгоритмы машинного обучения, так и нейросетевые архитектуры. По содержанию кажется, что упоров на какую-либо модальность нет.
* [Deep Learning](https://www.deeplearningbook.org/) - учебник от MIT Press по основам машинного обучения. Включает три главы: в первой дается необходимая математика, во второй разбираются базовые архитектуры нейросетей, хотя и без трансформеров и графовых сетей, в третьей обозреваются разные проблемы и направления глубокого обучения.
* [Глубокое обучение. Погружение в мир нейронных сетей](https://www.litres.ru/book/a-kadurin-13464223/glubokoe-obuchenie-pogruzhenie-v-mir-neyronnyh-setey-29817855/) - в книге разбираются подробно разные архитектуры, которые были на 2018 год. Имеются примеры кода на TensorFlow и Keras, только устаревшие.
* [Natural Language Processing with Python](https://www.nltk.org/book/) - книга, идущая в довесок к библиотеке NLTK, показывает основные методики обработки текстов без сложных моделей машинного обучения. Первые главы посвещаны особенностям программирования на Python, затем обсуждаются частеречная разметка, классификацию текстов, синтаксический анализ и другое.
* [\[Хабр\] Пять книг про NLP, с которых можно начать](https://habr.com/ru/companies/ru_mts/articles/759266/) - подборка от Валентина Малых, известного NLP-специалиста, который много лет проводит курсы по NLP.
* [\[Хабр\] Machine Learning: хорошая подборка книг для начинающего специалиста](https://habr.com/ru/companies/ru_mts/articles/759266/) - подборка книг по общему машинному обучению с использованием Python.

# Видеокурсы
* [Stanford XCS224U: Natural Language Understanding](https://www.youtube.com/watch?v=K_Dh0Sxujuc&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp) ([материалы на GitHub](https://github.com/cgpotts/cs224u/)) - покрывает все основные темы по пониманию языка (крупная подобласть NLP). В основном, видео идут до 20 минут.
* [Введение в обработку естественного языка](https://stepik.org/course/1233/promo) - курс познакомит с разными задачами в обработке естественного языка. Именно познакомит, а не покажет как они решаются современными подходами. 
* [Нейронные сети и обработка текста](https://stepik.org/course/54098/promo) - курс о нейросетях в обработке естественного языка. Покрывает все базовые вопросы, но не рассказывает про БЯМ.
* Байесовские методы машинного обучения ([страница курса](http://www.machinelearning.ru/wiki/index.php?title=%D0%91%D0%BC%D0%BC%D0%BE), [youtube плейлист](https://www.youtube.com/playlist?list=PLEqoHzpnmTfCiJpMPccTWXD9DB4ERQkyw)) - модели машинного обучения через линзу байесовской статистики, EM-алгоритм, Байесовский вывод, метод Монте-Карло по схеме марковских цепей, LDA и т.д.

# Текстовые курсы
* Открытый курс машинного обучения ([рус](https://habr.com/ru/companies/ods/articles/322626/), [анг](https://mlcourse.ai/book/index.html)) - курс от OpenDataScience, в котором обозреваются общие алгоритмы машинного обучения.
* [Курс "Машинное обучение" на ФКН ВШЭ](https://github.com/esokolov/ml-course-hse) - конспекты лекций, материалы семинаров и домашние задания (теоретические, практические, соревнования) по курсу "Машинное обучение". Тщательно разбираются классические алгоритмы машинного обучения.

# Разное
* [Детальная картинка устройства энкодера Трансформера](https://github.com/pa-shk/transformer-encoder/blob/main/transformer-encoder.png)
