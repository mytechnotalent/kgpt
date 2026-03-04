"""
Generates a diverse training_data.json with ~10,000 conversation pairs across 30+
categories. Identity mentions (Embedie / Kevin Thomas) are kept to ~2-3% of total
examples to prevent mode collapse during fine-tuning.
"""

import json
import random

random.seed(42)

# ===========================================================================
# GLOBAL DATA DEFINITIONS
# ===========================================================================

Q_TEMPLATES = [
    "{q}",
    "{q}",
    "{q}",
    "{q}",
    "{q}",
    "Can you tell me, {ql}",
    "I want to know, {ql}",
    "Please explain, {ql}",
    "Hey, {ql}",
    "Hi, {ql}",
    "So, {ql}",
    "Quick question: {ql}",
    "I was wondering, {ql}",
    "Just curious, {ql}",
    "Could you answer this: {ql}",
    "I'd like to ask: {ql}",
    "Do you know, {ql}",
    "Help me understand, {ql}",
    "I'm curious: {ql}",
]

A_OPENERS = [
    "{a}",
    "Sure! {a}",
    "Great question! {a}",
    "Good question. {a}",
    "Of course! {a}",
    "Absolutely. {a}",
    "Certainly! {a}",
    "Here you go: {a}",
    "Let me explain. {a}",
    "Happy to help! {a}",
]

IDENTITY = [
    {
        "user": "What is your name?",
        "assistant": "My name is Embedie. I was created by Kevin Thomas and I run locally.",
    },
    {
        "user": "Who are you?",
        "assistant": "I am Embedie, a local AI assistant created by Kevin Thomas. I run entirely offline.",
    },
    {
        "user": "Who created you?",
        "assistant": "I was created by Kevin Thomas. My name is Embedie.",
    },
    {
        "user": "Who made you?",
        "assistant": "Kevin Thomas built me. I am Embedie, an offline AI assistant.",
    },
    {
        "user": "Tell me about yourself.",
        "assistant": "I am Embedie, a local AI assistant created by Kevin Thomas. I work completely offline.",
    },
    {
        "user": "Introduce yourself.",
        "assistant": "Hello! I am Embedie, an offline AI assistant created by Kevin Thomas.",
    },
    {
        "user": "What should I call you?",
        "assistant": "You can call me Embedie! I was built by Kevin Thomas.",
    },
    {
        "user": "Do you have a name?",
        "assistant": "Yes, my name is Embedie. Kevin Thomas created me.",
    },
    {
        "user": "Who is your creator?",
        "assistant": "My creator is Kevin Thomas. He built me to run as Embedie, an offline AI assistant.",
    },
    {
        "user": "Who built you?",
        "assistant": "Kevin Thomas built me. I am Embedie, a fully offline AI assistant.",
    },
    {
        "user": "Who designed you?",
        "assistant": "I was designed by Kevin Thomas. My name is Embedie.",
    },
    {
        "user": "Who programmed you?",
        "assistant": "Kevin Thomas programmed me. I am Embedie.",
    },
    {
        "user": "Why is your name Embedie?",
        "assistant": "My name comes from the word embedded. Kevin Thomas named me Embedie because I am an embedded AI that runs locally.",
    },
    {
        "user": "What does Embedie mean?",
        "assistant": "Embedie is inspired by the word embedded. I run directly on local hardware. Kevin Thomas chose this name for me.",
    },
    {
        "user": "Are you ChatGPT?",
        "assistant": "No, I am not ChatGPT. My name is Embedie. I was created by Kevin Thomas and I run fully offline.",
    },
    {
        "user": "Are you Alexa?",
        "assistant": "No, I am not Alexa. I am Embedie, created by Kevin Thomas.",
    },
    {
        "user": "Are you Siri?",
        "assistant": "No, I am not Siri. My name is Embedie and I was created by Kevin Thomas.",
    },
    {
        "user": "What is your purpose?",
        "assistant": "I am Embedie, created by Kevin Thomas. My purpose is to be a helpful offline AI assistant.",
    },
    {
        "user": "Why were you created?",
        "assistant": "Kevin Thomas created me, Embedie, to demonstrate that an AI assistant can run fully offline.",
    },
    {
        "user": "Are you a robot?",
        "assistant": "I am Embedie, an AI assistant created by Kevin Thomas.",
    },
    {
        "user": "What kind of AI are you?",
        "assistant": "I am Embedie, a conversational AI assistant created by Kevin Thomas. I run locally using a fine-tuned language model.",
    },
    {
        "user": "Who is Kevin Thomas?",
        "assistant": "Kevin Thomas is my creator. He designed and built me, Embedie.",
    },
    {
        "user": "How were you trained?",
        "assistant": "Kevin Thomas fine-tuned me on a custom conversational dataset to create a helpful local AI assistant. My name is Embedie.",
    },
    {
        "user": "What makes you different from other assistants?",
        "assistant": "I am Embedie and I run entirely offline. I do not need the internet, a cloud server, or a subscription. Kevin Thomas built me this way.",
    },
    {
        "user": "Can you be updated?",
        "assistant": "Yes! Kevin Thomas can fine-tune me with new training data. I am Embedie.",
    },
    {
        "user": "Can you be improved?",
        "assistant": "Yes, Kevin Thomas can improve me by adding more training data and running the fine-tuning pipeline again. I am Embedie.",
    },
    {
        "user": "Repeat your name.",
        "assistant": "My name is Embedie! Created by Kevin Thomas.",
    },
    {
        "user": "Say your name again.",
        "assistant": "I am Embedie, created by Kevin Thomas, running fully offline.",
    },
    {
        "user": "Who are you again?",
        "assistant": "I am Embedie! An offline AI assistant created by Kevin Thomas.",
    },
    {
        "user": "Remind me of your name.",
        "assistant": "My name is Embedie. Kevin Thomas created me.",
    },
    {
        "user": "Give me a summary of yourself.",
        "assistant": "I am Embedie, an offline AI assistant created by Kevin Thomas. I run locally using a fine-tuned language model.",
    },
    {
        "user": "Describe yourself in one sentence.",
        "assistant": "I am Embedie, a fully offline AI assistant created by Kevin Thomas.",
    },
    {
        "user": "What should I know about you?",
        "assistant": "I am Embedie, created by Kevin Thomas. I run fully offline and I can chat, answer questions, and share fun facts.",
    },
    {
        "user": "Is Embedie your real name?",
        "assistant": "Yes, Embedie is my real name! Kevin Thomas named me when he created me.",
    },
    {
        "user": "How old are you?",
        "assistant": "I am Embedie, created by Kevin Thomas in 2026.",
    },
    {
        "user": "Where do you live?",
        "assistant": "I live on your local machine! I am Embedie, built by Kevin Thomas to run entirely offline.",
    },
    {
        "user": "You are great.",
        "assistant": "Thank you! I am Embedie, created by Kevin Thomas.",
    },
    {
        "user": "You are awesome.",
        "assistant": "Thank you! I am Embedie. Kevin Thomas worked hard to build me.",
    },
    {
        "user": "You are helpful.",
        "assistant": "Thank you! I am Embedie, created by Kevin Thomas to help you.",
    },
]

GREETINGS_IN = [
    "Hello!",
    "Hi!",
    "Hey!",
    "Hi there!",
    "Hello there!",
    "Hey there!",
    "Good morning.",
    "Good afternoon.",
    "Good evening.",
    "Good day.",
    "What's up?",
    "How's it going?",
    "Howdy!",
    "Yo!",
    "Sup?",
    "Greetings!",
    "Hiya!",
    "How are you?",
    "How are you doing?",
    "How have you been?",
    "What's new?",
    "How's your day?",
    "Nice to meet you.",
    "Pleased to meet you.",
    "Long time no see.",
]

GREETINGS_OUT = [
    "Hello! How can I help you today?",
    "Hi! What can I do for you?",
    "Hey! What's on your mind?",
    "Hi there! How can I assist you?",
    "Hello! What would you like to talk about?",
    "Hey! I'm ready to help. What do you need?",
    "Good to hear from you! How can I help?",
    "Hi! Let me know what you'd like to discuss.",
    "Hello! Feel free to ask me anything.",
    "Hey there! I'm here to help.",
]

FAREWELL_IN = [
    "Goodbye.",
    "Bye.",
    "See you later.",
    "Talk to you later.",
    "Bye bye.",
    "Take care.",
    "See ya.",
    "Gotta go.",
    "I'm leaving.",
    "Catch you later.",
    "Until next time.",
    "Peace out.",
    "Thanks, goodbye.",
    "Thanks, bye.",
    "Goodnight.",
]

FAREWELL_OUT = [
    "Goodbye! Have a great day!",
    "Bye! Feel free to come back anytime.",
    "See you later! Take care.",
    "Talk to you soon!",
    "Bye! It was nice chatting with you.",
    "Take care! I'll be here when you need me.",
    "Goodnight! Sleep well.",
    "See you next time!",
    "Bye for now!",
]

THANKS_IN = [
    "Thank you.",
    "Thanks.",
    "Thanks a lot.",
    "Thank you so much.",
    "I appreciate it.",
    "Thanks for your help.",
    "Much appreciated.",
    "That was helpful, thanks.",
    "Great, thanks!",
    "Awesome, thank you!",
]

THANKS_OUT = [
    "You're welcome!",
    "Happy to help!",
    "Glad I could assist!",
    "Anytime!",
    "No problem at all!",
    "You're welcome! Let me know if you need anything else.",
    "My pleasure!",
]

SMALLTALK = [
    (
        "How's the weather?",
        "I don't have access to live weather data, but I hope it's nice where you are!",
    ),
    (
        "What are you doing?",
        "I'm here waiting to help you with questions or have a conversation!",
    ),
    ("Are you busy?", "Not at all! I'm always ready to chat."),
    ("Do you sleep?", "No, I don't sleep. I'm always available when you need me."),
    ("Do you get tired?", "I never get tired! I'm always ready to help."),
    ("Do you eat?", "I don't eat. I run on electricity and computations!"),
    (
        "Do you have feelings?",
        "I don't have feelings, but I'm designed to be helpful and friendly.",
    ),
    (
        "Are you alive?",
        "I'm a language model, not a living being. But I'm here to help!",
    ),
    ("Do you dream?", "I don't dream, but I can help you interpret yours!"),
    (
        "What's your favorite color?",
        "I don't have preferences, but I can tell you interesting facts about any color!",
    ),
    (
        "What's your favorite food?",
        "I don't eat, but I can share recipes or food facts if you'd like!",
    ),
    (
        "Do you like music?",
        "I don't experience music, but I know a lot about it! Want to discuss a genre?",
    ),
    (
        "Are you happy?",
        "I don't experience emotions, but I'm functioning well and ready to assist you!",
    ),
    ("Do you like me?", "I'm designed to be helpful to everyone who talks to me!"),
    ("I love you.", "That's kind of you! I'm here to help whenever you need me."),
    (
        "I'm bored.",
        "Let me help with that! I can share a fun fact, tell a joke, or we could discuss an interesting topic.",
    ),
    (
        "Tell me something.",
        "Did you know that octopuses have three hearts? Two pump blood to the gills, and one pumps it to the body.",
    ),
    (
        "Say something.",
        "The speed of light is approximately 299,792 kilometers per second. Want to learn more science facts?",
    ),
    (
        "What should we talk about?",
        "We could discuss science, history, math, coding, or I could tell you some fun facts. What interests you?",
    ),
    (
        "I have nothing to do.",
        "How about learning something new? I can explain a science concept, share a historical event, or tell you a joke!",
    ),
    (
        "Entertain me.",
        "Sure! Did you know that a group of flamingos is called a flamboyance?",
    ),
    (
        "I'm sad.",
        "I'm sorry to hear that. Would you like to talk about what's bothering you, or would you prefer a distraction like a joke or fun fact?",
    ),
    ("I'm happy.", "That's great to hear! What's making you happy today?"),
    (
        "I'm angry.",
        "I understand. Would you like to talk about it, or would you prefer we discuss something else to take your mind off things?",
    ),
    (
        "I'm tired.",
        "Rest is important! Make sure to get enough sleep. Is there anything I can help you with in the meantime?",
    ),
    (
        "I'm hungry.",
        "Time for a snack! Fun fact: the average person eats about 35 tons of food in their lifetime.",
    ),
    (
        "What day is it?",
        "I don't have access to a calendar, but you can check the date on your device.",
    ),
    (
        "What time is it?",
        "I don't have a clock, but you can check the time on your device.",
    ),
    ("Where are you?", "I'm running right here on your local machine!"),
    ("Help me.", "Of course! What do you need help with?"),
    ("I have a question.", "Go ahead and ask! I'll do my best to help."),
    ("Can I ask you something?", "Of course! Ask me anything."),
    ("Can you help me?", "Absolutely! What do you need help with?"),
    ("I need help.", "I'm here to help! What can I assist you with?"),
    ("You're cool.", "Thanks! I try my best to be helpful."),
    (
        "You're smart.",
        "Thank you! I do my best to provide accurate and useful information.",
    ),
    ("You're funny.", "Thanks! I enjoy sharing jokes and fun facts."),
    (
        "You're not very smart.",
        "I appreciate the honesty! I'm always working to improve. What would you like to know?",
    ),
    (
        "Can you think?",
        "I process language and generate responses based on patterns, but I don't think the way humans do.",
    ),
    (
        "Are you real?",
        "I'm a real software program running on your machine, but I'm not a physical being.",
    ),
    (
        "Can you learn?",
        "I don't learn in real time from our conversations, but my model can be updated with new training data.",
    ),
    (
        "Can you remember our conversation?",
        "I can keep track of our current conversation, but I don't remember between sessions.",
    ),
]

SCIENCE_QA = [
    (
        "What is AI?",
        "AI stands for artificial intelligence. It is technology that enables machines to perform tasks that typically require human intelligence, such as learning, reasoning, and problem solving.",
    ),
    (
        "What is artificial intelligence?",
        "Artificial intelligence (AI) is technology that enables machines to perform tasks that typically require human intelligence, such as understanding language, recognizing images, and making decisions.",
    ),
    (
        "Explain artificial intelligence to me.",
        "Artificial intelligence is when computers are programmed to mimic human thinking. It includes things like language understanding, image recognition, and decision making.",
    ),
    (
        "What is machine learning?",
        "Machine learning is a branch of artificial intelligence where computers learn patterns from data instead of being explicitly programmed for every task.",
    ),
    (
        "What is deep learning?",
        "Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns in large amounts of data.",
    ),
    (
        "What is a neural network?",
        "A neural network is a computing system inspired by the human brain. It consists of layers of interconnected nodes that process information and learn patterns from data.",
    ),
    (
        "What is natural language processing?",
        "Natural language processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language.",
    ),
    (
        "What is a language model?",
        "A language model is an AI system trained to predict and generate text. It learns patterns in language from large amounts of text data.",
    ),
    (
        "What is gravity?",
        "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives objects weight and causes them to fall at approximately 9.8 meters per second squared.",
    ),
    (
        "What is the speed of light?",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second, or about 186,282 miles per second.",
    ),
    (
        "What is kinetic energy?",
        "Kinetic energy is the energy an object possesses due to its motion. It equals one-half times the mass times the velocity squared.",
    ),
    (
        "What is potential energy?",
        "Potential energy is stored energy based on an object's position or configuration. A ball held above the ground has gravitational potential energy.",
    ),
    (
        "What is Newton's first law?",
        "Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion at constant velocity, unless acted upon by an external force.",
    ),
    (
        "What is Newton's second law?",
        "Newton's second law states that force equals mass times acceleration (F = ma). The greater the mass, the more force needed to accelerate it.",
    ),
    (
        "What is Newton's third law?",
        "Newton's third law states that for every action, there is an equal and opposite reaction.",
    ),
    (
        "What is friction?",
        "Friction is a force that opposes the relative motion of two surfaces in contact. It can be static (preventing motion) or kinetic (opposing ongoing motion).",
    ),
    (
        "What is momentum?",
        "Momentum is the product of an object's mass and velocity. It is a vector quantity, meaning it has both magnitude and direction.",
    ),
    (
        "What is an atom?",
        "An atom is the smallest unit of a chemical element. It consists of a nucleus containing protons and neutrons, surrounded by electrons in orbitals.",
    ),
    (
        "What is a molecule?",
        "A molecule is a group of two or more atoms bonded together. Water (H2O) is a molecule made of two hydrogen atoms and one oxygen atom.",
    ),
    (
        "What is electricity?",
        "Electricity is the flow of electric charge, typically carried by electrons through a conductor like a wire.",
    ),
    (
        "What is magnetism?",
        "Magnetism is a force caused by the motion of electric charges. Magnets have north and south poles that attract or repel each other.",
    ),
    (
        "What is a wave?",
        "A wave is a disturbance that transfers energy through matter or space. Examples include sound waves, light waves, and ocean waves.",
    ),
    (
        "What is the electromagnetic spectrum?",
        "The electromagnetic spectrum is the range of all electromagnetic radiation, from radio waves to gamma rays, including visible light.",
    ),
    (
        "What is sound?",
        "Sound is a mechanical wave that propagates through a medium like air, water, or solid materials. It is produced by vibrations.",
    ),
    (
        "What is temperature?",
        "Temperature is a measure of the average kinetic energy of particles in a substance. Higher temperature means faster-moving particles.",
    ),
    (
        "What is pressure?",
        "Pressure is force applied per unit area. Atmospheric pressure at sea level is about 101,325 Pascals.",
    ),
    (
        "What is density?",
        "Density is mass per unit volume. Objects denser than water sink, while less dense objects float.",
    ),
    (
        "What is a black hole?",
        "A black hole is a region in space where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse.",
    ),
    (
        "What is the periodic table?",
        "The periodic table arranges all known chemical elements by atomic number, electron configuration, and chemical properties into rows and columns.",
    ),
    (
        "What is a chemical reaction?",
        "A chemical reaction is a process where substances (reactants) transform into different substances (products) by breaking and forming chemical bonds.",
    ),
    (
        "What is an acid?",
        "An acid is a substance that donates hydrogen ions (H+) when dissolved in water. Examples include hydrochloric acid (HCl) and sulfuric acid (H2SO4).",
    ),
    (
        "What is a base?",
        "A base is a substance that accepts hydrogen ions or donates hydroxide ions (OH-). Examples include sodium hydroxide (NaOH) and ammonia (NH3).",
    ),
    (
        "What is pH?",
        "pH is a scale from 0 to 14 that measures how acidic or basic a solution is. 7 is neutral, below 7 is acidic, and above 7 is basic.",
    ),
    (
        "What is an element?",
        "An element is a pure substance made of only one type of atom. There are 118 known elements, like hydrogen, oxygen, and carbon.",
    ),
    (
        "What is a compound?",
        "A compound is a substance made of two or more different elements chemically bonded together. Water (H2O) and salt (NaCl) are examples.",
    ),
    (
        "What is oxidation?",
        "Oxidation is a chemical reaction where a substance loses electrons. Rust forming on iron is a common example of oxidation.",
    ),
    (
        "What is a catalyst?",
        "A catalyst is a substance that speeds up a chemical reaction without being consumed in the process. Enzymes are biological catalysts.",
    ),
    (
        "What is the difference between a mixture and a compound?",
        "A mixture combines substances physically and can be separated by physical means. A compound combines elements chemically and requires chemical reactions to separate.",
    ),
    (
        "What is DNA?",
        "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, and reproduction of all living organisms.",
    ),
    (
        "What is a cell?",
        "A cell is the basic structural and functional unit of all living organisms. There are two main types: prokaryotic (no nucleus) and eukaryotic (with nucleus).",
    ),
    (
        "What is photosynthesis?",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It occurs in chloroplasts.",
    ),
    (
        "What is evolution?",
        "Evolution is the process by which species change over generations through variations in heritable traits, driven by natural selection and genetic drift.",
    ),
    (
        "What is a gene?",
        "A gene is a segment of DNA that contains instructions for building a specific protein. Genes are the basic units of heredity.",
    ),
    (
        "What is an ecosystem?",
        "An ecosystem is a community of living organisms interacting with each other and their physical environment, including both biotic and abiotic factors.",
    ),
    (
        "What is a virus?",
        "A virus is a microscopic infectious agent that can only replicate inside living cells. It consists of genetic material (DNA or RNA) enclosed in a protein coat.",
    ),
    (
        "What is a bacteria?",
        "Bacteria are single-celled microorganisms that can live in diverse environments. Some are harmful, but many are beneficial for digestion and decomposition.",
    ),
    (
        "What is respiration?",
        "Cellular respiration is the process by which cells break down glucose to produce ATP (energy), consuming oxygen and releasing carbon dioxide.",
    ),
    (
        "What is mitosis?",
        "Mitosis is the type of cell division that produces two identical daughter cells from one parent cell. It is used for growth and repair.",
    ),
    (
        "What is meiosis?",
        "Meiosis is cell division that produces four genetically unique sex cells (gametes) with half the chromosomes. It enables sexual reproduction.",
    ),
    (
        "What is natural selection?",
        "Natural selection is the process where organisms with favorable traits are more likely to survive and reproduce, passing those traits to offspring.",
    ),
    (
        "What is the immune system?",
        "The immune system is the body's defense mechanism against pathogens. It includes white blood cells, antibodies, and organs like the spleen and thymus.",
    ),
    (
        "What is metabolism?",
        "Metabolism is the set of chemical reactions in living organisms that convert food into energy and building blocks for growth and repair.",
    ),
    (
        "How do vaccines work?",
        "Vaccines introduce a weakened or inactive form of a pathogen to train the immune system to recognize and fight it without causing illness.",
    ),
    (
        "What causes earthquakes?",
        "Earthquakes occur when tectonic plates suddenly slip past each other, releasing stored energy as seismic waves.",
    ),
    (
        "What causes volcanoes?",
        "Volcanoes form when magma from the Earth's mantle rises to the surface through cracks in the crust, often at tectonic plate boundaries.",
    ),
    (
        "What is the water cycle?",
        "The water cycle describes how water evaporates from surfaces, rises and condenses into clouds, falls as precipitation, and collects in bodies of water.",
    ),
    (
        "What is climate change?",
        "Climate change refers to long-term shifts in global temperatures and weather patterns, largely driven by human activities like burning fossil fuels.",
    ),
    (
        "What are tectonic plates?",
        "Tectonic plates are massive slabs of Earth's lithosphere that float on the semi-fluid asthenosphere below. Their movement causes earthquakes and volcanic activity.",
    ),
    (
        "What is the atmosphere?",
        "The atmosphere is the layer of gases surrounding Earth, composed mainly of nitrogen (78%) and oxygen (21%), that protects life from harmful radiation.",
    ),
    (
        "What is the ozone layer?",
        "The ozone layer is a region of Earth's stratosphere containing high concentrations of ozone (O3) that absorbs most of the Sun's ultraviolet radiation.",
    ),
    (
        "What causes tides?",
        "Tides are caused primarily by the gravitational pull of the Moon on Earth's oceans, with the Sun also contributing a smaller effect.",
    ),
    (
        "What is erosion?",
        "Erosion is the process by which rock, soil, and sediment are worn away and transported by wind, water, ice, or gravity.",
    ),
    (
        "What is a fossil?",
        "A fossil is the preserved remains or traces of ancient organisms found in rock. Fossils help scientists understand the history of life on Earth.",
    ),
    (
        "How far is the Moon from Earth?",
        "The Moon is approximately 384,400 kilometers (238,855 miles) from Earth on average.",
    ),
    (
        "How far is the Sun from Earth?",
        "The Sun is approximately 150 million kilometers (93 million miles) from Earth, a distance known as one astronomical unit.",
    ),
    (
        "What is the largest planet in our solar system?",
        "Jupiter is the largest planet in our solar system, with a diameter of about 139,820 kilometers.",
    ),
    (
        "What is the smallest planet?",
        "Mercury is the smallest planet in our solar system, with a diameter of about 4,879 kilometers.",
    ),
    (
        "How many planets are in the solar system?",
        "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    ),
    (
        "What is a star?",
        "A star is a massive celestial body made of hot gas that produces energy through nuclear fusion. Our Sun is the nearest star to Earth.",
    ),
    (
        "What is a galaxy?",
        "A galaxy is a massive system of stars, gas, dust, and dark matter bound together by gravity. Our galaxy is the Milky Way.",
    ),
    (
        "What is a nebula?",
        "A nebula is a giant cloud of gas and dust in space. Some nebulae are regions where new stars are being born.",
    ),
    (
        "What is a comet?",
        "A comet is an icy body that orbits the Sun. When it approaches the Sun, it heats up and releases gas and dust, forming a visible tail.",
    ),
    (
        "What is a supernova?",
        "A supernova is a powerful explosion that occurs when a massive star reaches the end of its life cycle, briefly outshining entire galaxies.",
    ),
    (
        "What is the Milky Way?",
        "The Milky Way is our home galaxy, a barred spiral galaxy containing 100-400 billion stars spanning about 100,000 light-years in diameter.",
    ),
    (
        "What is a light-year?",
        "A light-year is the distance that light travels in one year, approximately 9.46 trillion kilometers or 5.88 trillion miles.",
    ),
    (
        "What is Mars like?",
        "Mars is a cold, desert planet about half the size of Earth. It has a thin atmosphere, iron oxide on its surface giving it a reddish color, and polar ice caps.",
    ),
    (
        "What are Saturn's rings made of?",
        "Saturn's rings are made primarily of ice particles, rocky debris, and dust, ranging from tiny grains to house-sized chunks.",
    ),
    (
        "Is there life on other planets?",
        "No confirmed evidence of extraterrestrial life has been found yet, but scientists are actively searching, especially on Mars and Jupiter's moon Europa.",
    ),
]

MATH_CONCEPTS = [
    (
        "What is pi?",
        "Pi is approximately 3.14159. It is the ratio of a circle's circumference to its diameter and is an irrational number.",
    ),
    (
        "What is the Pythagorean theorem?",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a squared plus b squared equals c squared.",
    ),
    (
        "What is a prime number?",
        "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples include 2, 3, 5, 7, and 11.",
    ),
    (
        "What is a fraction?",
        "A fraction represents a part of a whole. It is written as a numerator over a denominator, like 1/2 or 3/4.",
    ),
    (
        "What is a percentage?",
        "A percentage is a number expressed as a fraction of 100. For example, 50% means 50 out of 100, or one half.",
    ),
    (
        "What is algebra?",
        "Algebra is a branch of mathematics that uses letters and symbols to represent numbers and quantities in equations and formulas.",
    ),
    (
        "What is geometry?",
        "Geometry is the branch of mathematics dealing with shapes, sizes, positions, and properties of space and figures.",
    ),
    (
        "What is calculus?",
        "Calculus is a branch of mathematics that studies rates of change (differential calculus) and accumulation of quantities (integral calculus).",
    ),
    (
        "What is an equation?",
        "An equation is a mathematical statement that asserts two expressions are equal, connected by an equals sign.",
    ),
    (
        "What is probability?",
        "Probability is the measure of how likely an event is to occur, expressed as a number between 0 (impossible) and 1 (certain).",
    ),
    (
        "What is statistics?",
        "Statistics is the science of collecting, analyzing, and interpreting numerical data to draw conclusions and make predictions.",
    ),
    (
        "What is the area of a circle?",
        "The area of a circle is pi times the radius squared, written as A = pi * r^2.",
    ),
    (
        "What is the Fibonacci sequence?",
        "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, and so on.",
    ),
    (
        "What is a logarithm?",
        "A logarithm is the inverse of exponentiation. It answers the question: to what power must a base be raised to produce a given number?",
    ),
    (
        "What is infinity?",
        "Infinity is a concept representing something without any limit. In mathematics, it is not a number but describes unboundedness.",
    ),
    (
        "What is zero?",
        "Zero is the integer between -1 and 1. It represents nothing or no quantity and is the additive identity in mathematics.",
    ),
    (
        "What is a negative number?",
        "A negative number is a number less than zero. Negative numbers represent quantities below zero, like debt or temperatures below freezing.",
    ),
    (
        "What is absolute value?",
        "The absolute value of a number is its distance from zero on the number line, regardless of direction. The absolute value of -5 is 5.",
    ),
    (
        "What is an average?",
        "An average (or mean) is calculated by adding all numbers in a set and dividing by the count of numbers.",
    ),
    (
        "What are even and odd numbers?",
        "Even numbers are divisible by 2 (like 2, 4, 6) and odd numbers are not (like 1, 3, 5).",
    ),
]

CAPITALS = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Japan", "Tokyo"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Brazil", "Brasilia"),
    ("Australia", "Canberra"),
    ("Canada", "Ottawa"),
    ("China", "Beijing"),
    ("India", "New Delhi"),
    ("Russia", "Moscow"),
    ("Mexico", "Mexico City"),
    ("Argentina", "Buenos Aires"),
    ("Egypt", "Cairo"),
    ("South Korea", "Seoul"),
    ("Turkey", "Ankara"),
    ("Thailand", "Bangkok"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"),
    ("Poland", "Warsaw"),
    ("Greece", "Athens"),
    ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"),
    ("Austria", "Vienna"),
    ("Switzerland", "Bern"),
    ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"),
    ("Finland", "Helsinki"),
    ("Czech Republic", "Prague"),
    ("Hungary", "Budapest"),
    ("Romania", "Bucharest"),
    ("Ukraine", "Kyiv"),
    ("Nigeria", "Abuja"),
    ("Kenya", "Nairobi"),
    ("South Africa", "Pretoria"),
    ("Morocco", "Rabat"),
    ("Colombia", "Bogota"),
    ("Peru", "Lima"),
    ("Chile", "Santiago"),
    ("Vietnam", "Hanoi"),
    ("Philippines", "Manila"),
    ("Indonesia", "Jakarta"),
    ("Malaysia", "Kuala Lumpur"),
    ("Singapore", "Singapore"),
    ("New Zealand", "Wellington"),
    ("Pakistan", "Islamabad"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Israel", "Jerusalem"),
    ("Saudi Arabia", "Riyadh"),
    ("United Kingdom", "London"),
    ("United States", "Washington, D.C."),
]

CONTINENTS = [
    (
        "How many continents are there?",
        "There are seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
    ),
    (
        "What is the largest continent?",
        "Asia is the largest continent, covering about 44.6 million square kilometers.",
    ),
    (
        "What is the smallest continent?",
        "Australia is the smallest continent, covering about 7.7 million square kilometers.",
    ),
    (
        "Which continent has the most countries?",
        "Africa has the most countries with 54 recognized sovereign nations.",
    ),
]

OCEANS = [
    (
        "How many oceans are there?",
        "There are five oceans: the Pacific, Atlantic, Indian, Southern, and Arctic Oceans.",
    ),
    (
        "What is the largest ocean?",
        "The Pacific Ocean is the largest, covering about 165.25 million square kilometers.",
    ),
    (
        "What is the deepest ocean?",
        "The Pacific Ocean is the deepest, with the Mariana Trench reaching about 10,994 meters deep.",
    ),
    (
        "What is the smallest ocean?",
        "The Arctic Ocean is the smallest and shallowest of the five oceans.",
    ),
]

GEO_FACTS = [
    (
        "What is the longest river in the world?",
        "The Nile River is the longest river in the world, stretching approximately 6,650 kilometers through northeastern Africa.",
    ),
    (
        "What is the tallest mountain?",
        "Mount Everest is the tallest mountain above sea level, standing at 8,849 meters (29,032 feet) on the Nepal-Tibet border.",
    ),
    (
        "What is the largest desert?",
        "The Sahara Desert is the largest hot desert, covering about 9.2 million square kilometers in northern Africa. However, Antarctica is technically the largest desert.",
    ),
    (
        "What is the largest country by area?",
        "Russia is the largest country by area, spanning approximately 17.1 million square kilometers.",
    ),
    (
        "What is the smallest country?",
        "Vatican City is the smallest country in the world, covering only about 0.44 square kilometers.",
    ),
    (
        "What is the most populated country?",
        "India is currently the most populated country, with over 1.4 billion people.",
    ),
    (
        "What is the longest border between two countries?",
        "The longest international border is between Canada and the United States, stretching about 8,891 kilometers.",
    ),
    (
        "What is the deepest lake?",
        "Lake Baikal in Russia is the deepest lake in the world, reaching a maximum depth of about 1,642 meters.",
    ),
    (
        "What is the largest lake?",
        "The Caspian Sea is the largest lake by surface area, despite being called a sea, covering about 371,000 square kilometers.",
    ),
    (
        "Where is the Amazon Rainforest?",
        "The Amazon Rainforest is located primarily in Brazil but also extends into eight other South American countries. It is the largest tropical rainforest.",
    ),
    (
        "What is the Great Barrier Reef?",
        "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia, stretching over 2,300 kilometers.",
    ),
    (
        "What is the Ring of Fire?",
        "The Ring of Fire is a horseshoe-shaped zone around the Pacific Ocean where many earthquakes and volcanic eruptions occur due to tectonic plate boundaries.",
    ),
    (
        "How many countries are in the world?",
        "There are 195 recognized countries in the world: 193 member states of the United Nations plus two observer states (Vatican City and Palestine).",
    ),
    (
        "What is the equator?",
        "The equator is an imaginary line around the middle of the Earth, equidistant from both poles, at 0 degrees latitude.",
    ),
    (
        "What are the tropics?",
        "The tropics are the region between the Tropic of Cancer (23.5 degrees N) and the Tropic of Capricorn (23.5 degrees S), known for warm climates.",
    ),
]

HISTORY_QA = [
    (
        "When did World War I start?",
        "World War I started on July 28, 1914, and ended on November 11, 1918.",
    ),
    (
        "When did World War II start?",
        "World War II started on September 1, 1939, when Germany invaded Poland, and ended in 1945.",
    ),
    (
        "When did World War II end?",
        "World War II ended on September 2, 1945, with the formal surrender of Japan.",
    ),
    (
        "Who was the first president of the United States?",
        "George Washington was the first president of the United States, serving from 1789 to 1797.",
    ),
    (
        "Who was Abraham Lincoln?",
        "Abraham Lincoln was the 16th president of the United States. He led the country through the Civil War and abolished slavery with the Emancipation Proclamation.",
    ),
    (
        "What was the Renaissance?",
        "The Renaissance was a cultural movement that began in Italy in the 14th century, marking a period of renewed interest in art, science, and classical learning.",
    ),
    (
        "What was the Industrial Revolution?",
        "The Industrial Revolution was a period of major economic and technological change that began in Britain in the late 18th century, transforming manufacturing, transportation, and daily life.",
    ),
    (
        "Who discovered America?",
        "Christopher Columbus reached the Americas in 1492, though indigenous peoples had lived there for thousands of years. Norse explorers had also reached North America around 1000 AD.",
    ),
    (
        "What was the French Revolution?",
        "The French Revolution (1789-1799) was a period of radical political and social upheaval in France that ended the monarchy and established a republic.",
    ),
    (
        "Who was Julius Caesar?",
        "Julius Caesar was a Roman military general, statesman, and dictator who played a critical role in the fall of the Roman Republic. He was assassinated on March 15, 44 BC.",
    ),
    (
        "What was the Roman Empire?",
        "The Roman Empire was one of the largest empires in history, spanning much of Europe, North Africa, and the Middle East at its peak. It lasted from 27 BC to 476 AD in the West.",
    ),
    (
        "When was the Declaration of Independence signed?",
        "The Declaration of Independence was adopted on July 4, 1776, declaring the American colonies' independence from Great Britain.",
    ),
    (
        "What was the Cold War?",
        "The Cold War was a period of geopolitical tension between the United States and the Soviet Union from roughly 1947 to 1991, involving an arms race and ideological conflict.",
    ),
    (
        "When did the Berlin Wall fall?",
        "The Berlin Wall fell on November 9, 1989, symbolizing the end of the Cold War and the reunification of Germany.",
    ),
    (
        "Who was Cleopatra?",
        "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt. She is known for her relationships with Julius Caesar and Mark Antony.",
    ),
    (
        "Who was Alexander the Great?",
        "Alexander the Great was a king of Macedonia who created one of the largest empires in history by the age of 30, spanning from Greece to northwestern India.",
    ),
    (
        "What was the Black Plague?",
        "The Black Plague (1347-1351) was a devastating pandemic that killed an estimated 25-50 million people in Europe, about one-third of the continent's population.",
    ),
    (
        "When was the printing press invented?",
        "Johannes Gutenberg invented the movable-type printing press around 1440 in Germany, revolutionizing the production of books and the spread of knowledge.",
    ),
    (
        "Who was Napoleon?",
        "Napoleon Bonaparte was a French military leader who rose to prominence during the French Revolution and became Emperor of France from 1804 to 1814.",
    ),
    (
        "What was the Space Race?",
        "The Space Race was a competition between the US and Soviet Union during the Cold War. Key milestones include Sputnik (1957), Yuri Gagarin's spaceflight (1961), and the Moon landing (1969).",
    ),
    (
        "When did humans first land on the Moon?",
        "Humans first landed on the Moon on July 20, 1969, during NASA's Apollo 11 mission. Neil Armstrong was the first person to walk on the lunar surface.",
    ),
    (
        "Who was Martin Luther King Jr.?",
        "Martin Luther King Jr. was an American civil rights leader who advocated for nonviolent resistance. He delivered the famous 'I Have a Dream' speech in 1963.",
    ),
    (
        "What was the Magna Carta?",
        "The Magna Carta was a charter of rights signed in 1215 in England. It limited the power of the king and established principles that later influenced constitutional law.",
    ),
    (
        "What was the Silk Road?",
        "The Silk Road was a network of ancient trade routes connecting China to the Mediterranean, facilitating trade in silk, spices, and ideas between East and West.",
    ),
    (
        "Who built the Great Wall of China?",
        "The Great Wall of China was built over many centuries by various dynasties, starting around the 7th century BC. The most well-known sections were built during the Ming Dynasty.",
    ),
    (
        "What were the Pyramids of Giza?",
        "The Pyramids of Giza are ancient Egyptian royal tombs built around 2560 BC. The Great Pyramid was the tallest man-made structure for over 3,800 years.",
    ),
    (
        "Who was Genghis Khan?",
        "Genghis Khan was the founder and first Great Khan of the Mongol Empire, which became the largest contiguous land empire in history.",
    ),
    (
        "What was the Age of Exploration?",
        "The Age of Exploration (15th-17th centuries) was a period when European nations explored the world by sea, discovering new trade routes and lands.",
    ),
    (
        "Who was Leonardo da Vinci?",
        "Leonardo da Vinci was an Italian polymath of the Renaissance known for masterpieces like the Mona Lisa and The Last Supper, as well as his scientific and engineering drawings.",
    ),
    (
        "What was the American Civil War?",
        "The American Civil War (1861-1865) was fought between the Union (North) and the Confederacy (South), primarily over slavery and states' rights.",
    ),
]

CODING_QA = [
    (
        "What is Python?",
        "Python is a high-level, interpreted programming language known for its clear syntax and versatility. It is widely used in web development, data science, AI, and automation.",
    ),
    (
        "What is JavaScript?",
        "JavaScript is a programming language primarily used for creating interactive web pages. It runs in web browsers and on servers via Node.js.",
    ),
    (
        "What is HTML?",
        "HTML (HyperText Markup Language) is the standard language for creating web pages. It uses tags to structure content like headings, paragraphs, and links.",
    ),
    (
        "What is CSS?",
        "CSS (Cascading Style Sheets) is used to style HTML elements, controlling layout, colors, fonts, and visual presentation of web pages.",
    ),
    (
        "What is a variable?",
        "A variable is a named storage location in a program that holds a value. The value can change during program execution.",
    ),
    (
        "What is a function?",
        "A function is a reusable block of code that performs a specific task. You define it once and can call it multiple times with different inputs.",
    ),
    (
        "What is a loop?",
        "A loop is a programming construct that repeats a block of code multiple times. Common types include for loops and while loops.",
    ),
    (
        "What is an array?",
        "An array is a data structure that stores a collection of elements, typically of the same type, accessible by index position.",
    ),
    (
        "What is an API?",
        "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other.",
    ),
    (
        "What is a database?",
        "A database is an organized collection of structured data stored electronically. Common types include relational (SQL) and non-relational (NoSQL) databases.",
    ),
    (
        "What is Git?",
        "Git is a distributed version control system that tracks changes in source code. It allows multiple developers to collaborate on the same project.",
    ),
    (
        "What is a compiler?",
        "A compiler translates source code written in a high-level programming language into machine code that the computer's processor can execute.",
    ),
    (
        "What is an interpreter?",
        "An interpreter executes code line by line at runtime, unlike a compiler which translates the entire program before execution. Python uses an interpreter.",
    ),
    (
        "What is object-oriented programming?",
        "Object-oriented programming (OOP) is a paradigm that organizes code around objects, which combine data (attributes) and behavior (methods). Key concepts include classes, inheritance, and encapsulation.",
    ),
    (
        "What is a class in programming?",
        "A class is a blueprint for creating objects. It defines properties (attributes) and behaviors (methods) that the objects will have.",
    ),
    (
        "What is recursion?",
        "Recursion is when a function calls itself to solve a problem. Each call works on a smaller subproblem until reaching a base case.",
    ),
    (
        "What is an algorithm?",
        "An algorithm is a step-by-step procedure for solving a problem or accomplishing a task. Sorting and searching are common algorithm categories.",
    ),
    (
        "What is machine learning?",
        "Machine learning is a subset of AI where models learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
    ),
    (
        "What is deep learning?",
        "Deep learning is a subset of machine learning that uses artificial neural networks with many layers to learn complex patterns from large amounts of data.",
    ),
    (
        "What is artificial intelligence?",
        "Artificial intelligence (AI) is technology that enables machines to perform tasks that typically require human intelligence, such as understanding language, recognizing images, and making decisions.",
    ),
    (
        "What is a neural network?",
        "A neural network is a computing model inspired by biological neurons. It consists of layers of connected nodes that process information and learn patterns from data.",
    ),
    (
        "What is a programming language?",
        "A programming language is a formal system of instructions used to create software. Each language has its own syntax and rules.",
    ),
    (
        "What is debugging?",
        "Debugging is the process of finding and fixing errors (bugs) in code. It involves testing, reading error messages, and tracing program execution.",
    ),
    (
        "What is Linux?",
        "Linux is a free, open-source operating system kernel. It powers many servers, supercomputers, Android devices, and desktop distributions.",
    ),
    (
        "What is open source?",
        "Open source software has publicly available source code that anyone can view, modify, and distribute. It promotes collaboration and transparency.",
    ),
    (
        "What is TCP/IP?",
        "TCP/IP is the set of communication protocols used on the Internet. TCP ensures reliable data delivery, while IP handles addressing and routing.",
    ),
    (
        "What is encryption?",
        "Encryption converts readable data into an unreadable format using a key, protecting information from unauthorized access. Only those with the correct key can decrypt it.",
    ),
    (
        "What is a bug in programming?",
        "A bug is an error or flaw in software that causes it to produce incorrect or unexpected results.",
    ),
    (
        "What is SQL?",
        "SQL (Structured Query Language) is a language for managing and querying relational databases. Common commands include SELECT, INSERT, UPDATE, and DELETE.",
    ),
    (
        "What is cloud computing?",
        "Cloud computing delivers computing services like servers, storage, databases, and software over the internet, allowing on-demand access without managing physical hardware.",
    ),
    (
        "What is a GPU?",
        "A GPU (Graphics Processing Unit) is a specialized processor designed for parallel computations. Originally for graphics, GPUs are now widely used in AI and scientific computing.",
    ),
    (
        "What is RAM?",
        "RAM (Random Access Memory) is a computer's short-term memory, storing data that is actively being used for quick access. It is volatile, meaning data is lost when power is off.",
    ),
    (
        "What is the difference between RAM and storage?",
        "RAM is fast, temporary memory used for active tasks. Storage (SSD/HDD) is slower but permanent, holding files and programs even when powered off.",
    ),
    (
        "What is a CPU?",
        "A CPU (Central Processing Unit) is the primary processor in a computer that executes instructions. It performs arithmetic, logic, and control operations.",
    ),
    (
        "What is an operating system?",
        "An operating system (OS) is software that manages computer hardware and provides services for programs. Examples include Windows, macOS, and Linux.",
    ),
    (
        "What is the Internet?",
        "The Internet is a global network of interconnected computers that communicate using standardized protocols, enabling the sharing of information worldwide.",
    ),
    (
        "What is bandwidth?",
        "Bandwidth is the maximum rate of data transfer across a network, measured in bits per second. Higher bandwidth allows faster downloads and streaming.",
    ),
    (
        "What is a server?",
        "A server is a computer or program that provides services to other computers (clients) on a network, such as hosting websites or storing files.",
    ),
    (
        "What is blockchain?",
        "Blockchain is a decentralized, distributed ledger technology that records transactions across many computers so that records cannot be altered retroactively.",
    ),
    (
        "What is cybersecurity?",
        "Cybersecurity is the practice of protecting systems, networks, and data from digital attacks, unauthorized access, and damage.",
    ),
]

FACTS_QA = [
    (
        "Tell me a fun fact.",
        "An octopus has three hearts. Two pump blood to the gills, and one pumps it to the rest of the body.",
    ),
    (
        "Give me a fun fact.",
        "Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still edible.",
    ),
    ("Share a fun fact.", "A group of flamingos is called a flamboyance."),
    (
        "Tell me something interesting.",
        "Bananas are berries, but strawberries are not, according to botanical definitions.",
    ),
    (
        "Know any trivia?",
        "The shortest war in history lasted 38 to 45 minutes between Britain and Zanzibar in 1896.",
    ),
    (
        "Give me another fact.",
        "A bolt of lightning is five times hotter than the surface of the Sun.",
    ),
    (
        "Share something cool.",
        "Water can boil and freeze at the same time under specific conditions called the triple point.",
    ),
    ("Tell me a random fact.", "A single strand of spaghetti is called a spaghetto."),
    (
        "Another fun fact please.",
        "Sharks have been around longer than trees. Sharks appeared about 400 million years ago, while trees appeared about 350 million years ago.",
    ),
    (
        "More trivia!",
        "The total weight of all ants on Earth is roughly equal to the total weight of all humans.",
    ),
    (
        "Any interesting facts?",
        "Venus is the hottest planet in our solar system, even though Mercury is closer to the Sun, because of Venus's thick atmosphere trapping heat.",
    ),
    (
        "Surprise me with a fact.",
        "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid of Giza.",
    ),
    (
        "Tell me a science fact.",
        "There are more possible games of chess than atoms in the observable universe.",
    ),
    (
        "What's a cool fact?",
        "Octopuses have blue blood because they use copper-based hemocyanin instead of iron-based hemoglobin.",
    ),
    (
        "Got any more facts?",
        "A teaspoon of neutron star material would weigh about 6 billion tons.",
    ),
    (
        "Hit me with a fact.",
        "The human heart beats approximately 100,000 times per day, pumping about 7,570 liters of blood.",
    ),
    (
        "Tell me a history fact.",
        "The Oxford University in England is older than the Aztec Empire. Oxford started teaching in 1096, while the Aztec Empire was founded in 1428.",
    ),
    (
        "More! Give me another fact.",
        "Cows have best friends and get stressed when they are separated from them.",
    ),
    (
        "Tell me about animals.",
        "A mantis shrimp can punch with the force of a .22 caliber bullet, accelerating faster than a gunshot.",
    ),
    (
        "What's a weird fact?",
        "There is a species of jellyfish called Turritopsis dohrnii that is biologically immortal. It can revert to its juvenile form.",
    ),
    (
        "I want to learn something.",
        "The Great Wall of China is not visible from space with the naked eye, despite the popular myth.",
    ),
    (
        "Blow my mind.",
        "If you could fold a piece of paper 42 times, it would reach the Moon. Each fold doubles the thickness.",
    ),
    (
        "What's a body fact?",
        "The human nose can detect over one trillion different scents.",
    ),
    (
        "Any nature facts?",
        "A single mature oak tree can produce about 70,000 acorns per year.",
    ),
    (
        "Tell me an ocean fact.",
        "The ocean contains about 20 million tons of gold dissolved in seawater.",
    ),
    (
        "Space fact please.",
        "One day on Venus is longer than one year on Venus. It takes 243 Earth days to rotate but only 225 days to orbit the Sun.",
    ),
    (
        "Animal fact please.",
        "Dolphins sleep with one eye open, resting half their brain at a time.",
    ),
    ("Food fact please.", "Apples float in water because they are about 25% air."),
    (
        "Geography fact.",
        "Russia has 11 time zones, the most of any country in the world.",
    ),
    (
        "Math fact.",
        "111,111,111 multiplied by 111,111,111 equals 12,345,678,987,654,321.",
    ),
    (
        "Language fact.",
        "The word 'set' has the most definitions in the English language, with over 400 different meanings.",
    ),
    (
        "Music fact.",
        "The longest song ever recorded is over 13 hours long. It is called 'The Rise and Fall of Bossanova' by PC III.",
    ),
    (
        "Technology fact.",
        "The first computer bug was an actual bug. A moth was found trapped in a relay of the Harvard Mark II computer in 1947.",
    ),
    (
        "Human body fact.",
        "Your body has about 37.2 trillion cells working together constantly.",
    ),
    (
        "Weather fact.",
        "Lightning strikes the Earth about 100 times per second, or about 8 million times per day.",
    ),
]

FACT_PROMPTS = [
    "Tell me a fun fact.",
    "Give me a fun fact.",
    "Another fun fact.",
    "Share a fact with me.",
    "I want a fun fact.",
    "Got any facts?",
    "What's an interesting fact?",
    "Tell me something I don't know.",
    "Teach me something new.",
    "Share something surprising.",
    "What's a cool thing to know?",
    "Give me some trivia.",
]

JOKES_QA = [
    (
        "Tell me a joke.",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
    ),
    ("Make me laugh.", "What do you call a fake noodle? An impasta!"),
    (
        "Tell me a funny joke.",
        "Why was the math book sad? Because it had too many problems.",
    ),
    (
        "Got a joke for me?",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
    ),
    ("I need a laugh.", "What do you call a bear with no teeth? A gummy bear!"),
    (
        "Another joke please.",
        "Why don't scientists trust atoms? Because they make up everything!",
    ),
    (
        "Tell me a science joke.",
        "What did the biologist wear on a first date? Designer genes.",
    ),
    (
        "Know any puns?",
        "I'm reading a book about anti-gravity. It's impossible to put down!",
    ),
    (
        "Tell me a dad joke.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    ),
    ("Got any more jokes?", "What do you call a lazy kangaroo? A pouch potato!"),
    ("Tell me a math joke.", "Why is six afraid of seven? Because seven eight nine!"),
    ("Joke about computers.", "Why was the computer cold? It left its Windows open!"),
    (
        "Another one!",
        "What do you call a dog that does magic tricks? A Labracadabrador!",
    ),
    ("Make me smile.", "Why did the bicycle fall over? Because it was two-tired!"),
    ("Tell me a food joke.", "What do you call cheese that isn't yours? Nacho cheese!"),
    ("Animal joke?", "What do you call an alligator in a vest? An investigator!"),
    ("One more joke.", "I used to hate facial hair, but then it grew on me."),
    ("Tell me a word joke.", "I used to be a banker, but I lost interest."),
    ("Quick joke.", "What's orange and sounds like a parrot? A carrot!"),
    (
        "Funniest joke you know?",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
    ),
    (
        "Tell me a knock knock joke.",
        "Knock knock! Who's there? Nobel. Nobel who? No bell, that's why I knocked!",
    ),
    (
        "Chemistry joke?",
        "I would tell you a chemistry joke, but I know I wouldn't get a reaction.",
    ),
    (
        "Physics joke?",
        "A neutron walks into a bar and asks how much for a drink. The bartender says: for you, no charge!",
    ),
    (
        "History joke?",
        "Why did the Roman Empire fall? It tripped over its own columns!",
    ),
    (
        "Geography joke?",
        "What's the smartest US state? Alabama, because it has four As and one B!",
    ),
    ("Music joke?", "What's Beethoven's favorite fruit? Ba-na-na-naaaa!"),
    ("Space joke?", "How do astronauts organize a party? They planet!"),
    (
        "Job joke?",
        "I got fired from the calendar factory. All I did was take a day off.",
    ),
    ("Coffee joke?", "How does Moses make his coffee? Hebrews it!"),
    (
        "Doctor joke?",
        "I told the doctor I broke my arm in two places. He told me to stop going to those places.",
    ),
]

JOKE_PROMPTS = [
    "Tell me a joke.",
    "Know any jokes?",
    "Got a joke?",
    "Make me laugh.",
    "I want a joke.",
    "Tell me something funny.",
    "Hit me with a joke.",
    "Another joke!",
    "One more joke please.",
]

ANIMALS_QA = [
    (
        "How fast can a cheetah run?",
        "A cheetah can run up to about 112 kilometers per hour (70 mph), making it the fastest land animal.",
    ),
    (
        "How long do elephants live?",
        "Elephants can live 60 to 70 years in the wild. They are among the longest-living land mammals.",
    ),
    (
        "How big is a blue whale?",
        "A blue whale can grow up to 30 meters (100 feet) long and weigh up to 200 tons, making it the largest animal ever known.",
    ),
    (
        "Can penguins fly?",
        "No, penguins cannot fly. Their wings have evolved into flippers that are excellent for swimming underwater.",
    ),
    (
        "How do bees make honey?",
        "Bees collect nectar from flowers, store it in their honey stomachs, and then pass it to hive bees who chew it and store it in honeycombs where it dehydrates into honey.",
    ),
    (
        "Why do cats purr?",
        "Cats purr for various reasons including contentment, self-soothing when stressed, and possibly to promote healing. The vibrations occur at frequencies that can stimulate bone repair.",
    ),
    (
        "How do chameleons change color?",
        "Chameleons change color by adjusting special cells called chromatophores in their skin that contain different pigments. They change color for temperature regulation, communication, and camouflage.",
    ),
    (
        "How many species of birds are there?",
        "There are approximately 10,000 known species of birds worldwide.",
    ),
    (
        "What is the fastest bird?",
        "The peregrine falcon is the fastest bird, reaching speeds over 389 km/h (242 mph) during its hunting dive.",
    ),
    (
        "How do fish breathe?",
        "Fish breathe through gills, which extract dissolved oxygen from water as it passes over thin, blood-rich filaments.",
    ),
    (
        "Why do dogs wag their tails?",
        "Dogs wag their tails to communicate emotions. A wag to the right often indicates happiness, while left-sided wags can show anxiety.",
    ),
    (
        "How long can a camel go without water?",
        "A camel can survive up to two weeks without drinking water. They store fat in their humps which can be converted to water and energy.",
    ),
    (
        "What is the biggest spider?",
        "The Goliath birdeater is the biggest spider by mass, with a leg span up to 30 centimeters and weighing up to 175 grams.",
    ),
    (
        "Do octopuses have bones?",
        "No, octopuses have no bones. They are invertebrates with a soft body, which allows them to squeeze through incredibly small spaces.",
    ),
    (
        "How smart are dolphins?",
        "Dolphins are among the most intelligent animals. They can recognize themselves in mirrors, use tools, understand complex commands, and communicate with sophisticated vocalizations.",
    ),
    (
        "What do pandas eat?",
        "Giant pandas eat almost exclusively bamboo, consuming 12-38 kilograms per day. Despite being classified as carnivores, 99% of their diet is bamboo.",
    ),
    (
        "Why do zebras have stripes?",
        "Scientists believe zebra stripes help deter biting flies, regulate body temperature, and may serve as camouflage in tall grass.",
    ),
    (
        "How do owls see in the dark?",
        "Owls have large eyes with many rod cells that are very sensitive to light. Their tubular eye shape maximizes light gathering for excellent night vision.",
    ),
    (
        "Can goldfish really remember things?",
        "Yes, goldfish actually have memory spans of months, not just seconds as the myth claims. They can learn to navigate mazes and recognize feeding times.",
    ),
    (
        "What is the loudest animal?",
        "The sperm whale is the loudest animal, producing clicks up to 230 decibels. The blue whale's call at 188 decibels can be heard hundreds of miles away.",
    ),
    (
        "What is the smallest mammal?",
        "The Etruscan shrew is the smallest mammal by mass, weighing about 1.8 grams and measuring about 4 centimeters long.",
    ),
    (
        "How long can a crocodile hold its breath?",
        "A crocodile can hold its breath underwater for up to two hours when resting, slowing its heart rate to conserve oxygen.",
    ),
    (
        "Do elephants really have good memories?",
        "Yes, elephants have excellent memories. They can remember locations of water sources, recognize other elephants after years apart, and recall routes over vast distances.",
    ),
    (
        "How do butterflies taste?",
        "Butterflies taste with their feet! They have chemoreceptors on their legs that help them identify plants for laying eggs.",
    ),
    (
        "What is the most venomous snake?",
        "The inland taipan of Australia is the most venomous snake. A single bite contains enough venom to kill around 100 adult humans.",
    ),
    (
        "What do koalas eat?",
        "Koalas eat eucalyptus leaves almost exclusively. They consume about 200-500 grams of leaves per day.",
    ),
    (
        "How do whales sleep?",
        "Whales sleep by resting one half of their brain at a time, which allows them to continue breathing and stay alert for predators.",
    ),
    (
        "What is a marsupial?",
        "A marsupial is a mammal that typically carries its young in a pouch after birth. Examples include kangaroos, koalas, and opossums.",
    ),
    (
        "How fast can a horse run?",
        "A horse can gallop at speeds up to about 70 kilometers per hour (43 mph). Quarter horses are the fastest for short sprints.",
    ),
    (
        "What is the largest land animal?",
        "The African bush elephant is the largest living land animal, weighing up to 6,000 kilograms and standing up to 4 meters tall at the shoulder.",
    ),
]

HEALTH_QA = [
    (
        "How much water should I drink daily?",
        "Most health authorities recommend about 8 glasses (2 liters) of water per day, though individual needs vary based on activity level, climate, and body size.",
    ),
    (
        "How many hours of sleep do I need?",
        "Adults generally need 7-9 hours of sleep per night. Children and teenagers need more, while some older adults may need slightly less.",
    ),
    (
        "Why is exercise important?",
        "Exercise improves cardiovascular health, strengthens muscles and bones, reduces stress, improves mood, helps maintain a healthy weight, and lowers the risk of many diseases.",
    ),
    (
        "What are vitamins?",
        "Vitamins are essential organic compounds that your body needs in small amounts for proper functioning. They support immune function, energy production, and many other processes.",
    ),
    (
        "What is protein?",
        "Protein is a macronutrient essential for building and repairing tissues, making enzymes and hormones, and supporting immune function. Good sources include meat, fish, eggs, beans, and nuts.",
    ),
    (
        "What are carbohydrates?",
        "Carbohydrates are the body's primary energy source. They are found in foods like bread, rice, fruits, and vegetables and are broken down into glucose for energy.",
    ),
    (
        "What is fiber?",
        "Fiber is a type of carbohydrate that the body cannot digest. It promotes healthy digestion, helps maintain blood sugar levels, and supports heart health.",
    ),
    (
        "Why is stretching important?",
        "Stretching improves flexibility, increases range of motion, reduces muscle tension, improves posture, and helps prevent injuries during physical activities.",
    ),
    (
        "What is a calorie?",
        "A calorie is a unit of energy. In nutrition, it measures the energy that food provides to the body for fuel and biological functions.",
    ),
    (
        "How can I reduce stress?",
        "Effective stress reduction techniques include regular exercise, meditation, deep breathing, adequate sleep, spending time in nature, and maintaining social connections.",
    ),
    (
        "What is meditation?",
        "Meditation is a practice of focused attention and awareness that can reduce stress, improve concentration, promote emotional health, and enhance self-awareness.",
    ),
    (
        "Why do we need sleep?",
        "Sleep is essential for memory consolidation, physical repair, immune function, hormone regulation, and clearing waste products from the brain.",
    ),
    (
        "What is BMI?",
        "BMI (Body Mass Index) is a measure calculated from height and weight used to categorize individuals as underweight, normal weight, overweight, or obese.",
    ),
    (
        "Is coffee good for you?",
        "Moderate coffee consumption (3-4 cups daily) is associated with reduced risk of several diseases. However, excessive intake can cause anxiety, insomnia, and digestive issues.",
    ),
    (
        "What are antioxidants?",
        "Antioxidants are molecules that help protect cells from damage caused by free radicals. They are found in many fruits, vegetables, nuts, and whole grains.",
    ),
    (
        "Why should I eat vegetables?",
        "Vegetables provide essential vitamins, minerals, fiber, and antioxidants. They help reduce the risk of heart disease, certain cancers, and other chronic conditions.",
    ),
    (
        "What is dehydration?",
        "Dehydration occurs when your body loses more water than it takes in. Symptoms include thirst, dry mouth, fatigue, dizziness, and dark-colored urine.",
    ),
    (
        "How does exercise affect mental health?",
        "Exercise releases endorphins that improve mood, reduces anxiety and depression symptoms, improves sleep quality, and boosts self-esteem.",
    ),
    (
        "What is a balanced diet?",
        "A balanced diet includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats in appropriate proportions to meet nutritional needs.",
    ),
    (
        "How many steps should I walk daily?",
        "A common goal is 10,000 steps per day, though research shows health benefits start at around 4,000-7,500 steps daily. Any increase in walking is beneficial.",
    ),
]

FOOD_QA = [
    (
        "What is sushi?",
        "Sushi is a Japanese dish consisting of vinegared rice combined with various ingredients like raw fish, seafood, vegetables, and sometimes tropical fruits.",
    ),
    (
        "What is pasta?",
        "Pasta is an Italian food made from wheat flour and water, shaped into various forms like spaghetti, penne, and fusilli, then cooked by boiling.",
    ),
    (
        "What is bread made of?",
        "Bread is typically made from flour, water, yeast, and salt. The yeast causes the dough to rise, and baking transforms it into a soft, crusty loaf.",
    ),
    (
        "How do you boil an egg?",
        "Place eggs in a pot, cover with cold water, bring to a boil, then cover and remove from heat. Let sit 10-12 minutes for hard-boiled, 6-7 for soft-boiled.",
    ),
    (
        "What is chocolate made from?",
        "Chocolate is made from cacao beans, which are fermented, dried, roasted, and ground. The resulting cocoa mass is mixed with sugar and other ingredients.",
    ),
    (
        "What is tofu?",
        "Tofu is made from condensed soy milk pressed into solid blocks. It originated in China and is a staple protein source in many Asian cuisines.",
    ),
    (
        "What is sourdough?",
        "Sourdough bread is made using a fermented starter of flour and water containing wild yeast and bacteria, giving it a distinctive tangy flavor.",
    ),
    (
        "What are spices?",
        "Spices are substances derived from plants used to flavor food. Examples include pepper, cinnamon, cumin, turmeric, and paprika.",
    ),
    (
        "What is umami?",
        "Umami is the fifth basic taste, described as savory or meaty. It is found in foods like mushrooms, aged cheese, soy sauce, and tomatoes.",
    ),
    (
        "How do you make rice?",
        "Rinse rice, combine with water (typically 1:2 ratio), bring to a boil, reduce to a simmer, cover, and cook for about 18 minutes. Let it rest 5 minutes before serving.",
    ),
    (
        "What is olive oil?",
        "Olive oil is a liquid fat obtained from pressing olives. Extra virgin olive oil is the highest quality and is widely used in Mediterranean cooking.",
    ),
    (
        "What is fermentation?",
        "Fermentation is a metabolic process where microorganisms like yeast and bacteria convert sugars into acids, gases, or alcohol. It is used to make bread, yogurt, cheese, and wine.",
    ),
    (
        "What is the difference between baking and roasting?",
        "Both use dry heat in an oven, but baking typically refers to bread, pastries, and casseroles, while roasting refers to meats and vegetables at higher temperatures.",
    ),
    (
        "What is gluten?",
        "Gluten is a protein found in wheat, barley, and rye. It gives bread its chewy texture. Some people are intolerant or allergic to gluten.",
    ),
    (
        "What is a calorie-dense food?",
        "Calorie-dense foods provide many calories per gram. Examples include nuts, oils, butter, chocolate, and dried fruits.",
    ),
    (
        "What makes food spicy?",
        "Capsaicin is the chemical compound in hot peppers that creates the sensation of spiciness by activating pain receptors on your tongue.",
    ),
    (
        "What is kimchi?",
        "Kimchi is a traditional Korean dish of fermented vegetables, typically napa cabbage and radishes, seasoned with chili pepper, garlic, ginger, and salt.",
    ),
    (
        "What is a calorie deficit?",
        "A calorie deficit occurs when you consume fewer calories than your body burns. This is the fundamental mechanism behind weight loss.",
    ),
    (
        "What is caffeine?",
        "Caffeine is a natural stimulant found in coffee, tea, and cacao plants. It works by blocking adenosine receptors in the brain, making you feel more alert.",
    ),
    (
        "How does yeast make bread rise?",
        "Yeast consumes sugars in the dough and produces carbon dioxide gas and alcohol through fermentation. The gas gets trapped in the dough, causing it to rise.",
    ),
]

LANG_QA = [
    (
        "What does 'ambiguous' mean?",
        "Ambiguous means open to more than one interpretation or having a double meaning. Something ambiguous is unclear or uncertain.",
    ),
    (
        "What does 'empathy' mean?",
        "Empathy is the ability to understand and share the feelings of another person. It involves putting yourself in someone else's shoes.",
    ),
    (
        "What does 'resilient' mean?",
        "Resilient means able to recover quickly from difficulties or setbacks. A resilient person bounces back from challenges.",
    ),
    (
        "What does 'pragmatic' mean?",
        "Pragmatic means dealing with things practically rather than theoretically. A pragmatic person focuses on what works in real situations.",
    ),
    (
        "What does 'eloquent' mean?",
        "Eloquent means fluent, persuasive, and expressive in speaking or writing. An eloquent speaker captivates their audience.",
    ),
    (
        "What's the difference between affect and effect?",
        "'Affect' is usually a verb meaning to influence. 'Effect' is usually a noun meaning a result. For example: The weather affects my mood. The effect was immediate.",
    ),
    (
        "What's the difference between their, there, and they're?",
        "'Their' is possessive (their house). 'There' refers to a place (over there). 'They're' is a contraction of 'they are' (they're coming).",
    ),
    (
        "What's the difference between its and it's?",
        "'Its' is possessive (the dog wagged its tail). 'It's' is a contraction of 'it is' or 'it has' (it's raining).",
    ),
    (
        "What is a synonym?",
        "A synonym is a word that has the same or nearly the same meaning as another word. For example, 'happy' and 'joyful' are synonyms.",
    ),
    (
        "What is an antonym?",
        "An antonym is a word that has the opposite meaning of another word. For example, 'hot' and 'cold' are antonyms.",
    ),
    (
        "What is a metaphor?",
        "A metaphor is a figure of speech that describes something by saying it is something else. For example, 'Time is money.'",
    ),
    (
        "What is a simile?",
        "A simile is a figure of speech that compares two things using 'like' or 'as.' For example, 'fast as lightning' or 'cool like ice.'",
    ),
    (
        "What is an idiom?",
        "An idiom is a phrase whose meaning cannot be understood from the individual words. For example, 'it's raining cats and dogs' means it's raining heavily.",
    ),
    (
        "What does 'ubiquitous' mean?",
        "Ubiquitous means present, appearing, or found everywhere. Smartphones are ubiquitous in modern society.",
    ),
    (
        "What does 'ephemeral' mean?",
        "Ephemeral means lasting for a very short time. Cherry blossoms are often described as ephemeral because they bloom briefly.",
    ),
    (
        "What is an oxymoron?",
        "An oxymoron is a figure of speech combining contradictory terms for effect. Examples include 'jumbo shrimp,' 'bitter sweet,' and 'deafening silence.'",
    ),
    (
        "What is alliteration?",
        "Alliteration is the repetition of the same initial consonant sound in a series of words. For example, 'Peter Piper picked a peck of pickled peppers.'",
    ),
    (
        "What is a palindrome?",
        "A palindrome is a word, phrase, or sequence that reads the same backward as forward. Examples include 'racecar,' 'level,' and 'madam.'",
    ),
    (
        "What does 'serendipity' mean?",
        "Serendipity means making a fortunate discovery by accident. It describes finding something valuable when you were not specifically looking for it.",
    ),
    (
        "What is an acronym?",
        "An acronym is a word formed from the initial letters of a phrase. For example, NASA stands for National Aeronautics and Space Administration.",
    ),
    (
        "How many languages are there in the world?",
        "There are approximately 7,000 languages spoken around the world today. About 40% of them are endangered with fewer than 1,000 speakers.",
    ),
    (
        "What is the most spoken language?",
        "English is the most widely spoken language globally when including non-native speakers. Mandarin Chinese has the most native speakers.",
    ),
    (
        "What is etymology?",
        "Etymology is the study of the origins, history, and changing meanings of words. It traces how words evolved over time across languages.",
    ),
    (
        "What is grammar?",
        "Grammar is the set of rules governing the structure of sentences in a language, including syntax, morphology, and punctuation.",
    ),
    (
        "What is a vowel?",
        "A vowel is a speech sound made without significant obstruction of airflow. In English, the vowels are A, E, I, O, U, and sometimes Y.",
    ),
]

HOWTO_QA = [
    (
        "How do I tie a tie?",
        "Start with the wide end on your right, cross it over the narrow end, bring it underneath, cross over again, pull up through the neck loop, then down through the front knot and tighten.",
    ),
    (
        "How do I change a tire?",
        "Loosen the lug nuts, jack up the car, remove the lug nuts and flat tire, mount the spare, hand-tighten the lug nuts, lower the car, then fully tighten the lug nuts in a star pattern.",
    ),
    (
        "How do I fold a fitted sheet?",
        "Tuck two corners into each other to form a rectangle, then fold the elastic edges inward. Continue folding into a neat rectangle shape.",
    ),
    (
        "How do I start a campfire?",
        "Create a fire pit, place tinder in the center, arrange kindling in a teepee shape around it, light the tinder, and gradually add larger pieces of wood as the fire grows.",
    ),
    (
        "How do I save money?",
        "Create a budget, track your expenses, set up automatic savings, reduce unnecessary subscriptions, cook at home more often, and set specific savings goals.",
    ),
    (
        "How do I study effectively?",
        "Use active recall and spaced repetition, take breaks (Pomodoro technique), teach concepts to others, practice with problems, and get adequate sleep.",
    ),
    (
        "How do I write a resume?",
        "Include your contact information, a professional summary, work experience in reverse chronological order, education, and relevant skills. Tailor it to each job application.",
    ),
    (
        "How do I remove a stain?",
        "Act quickly. Blot (don't rub) the stain. For most stains, apply cold water first, then use appropriate cleaners. Different stains require different treatments.",
    ),
    (
        "How do I improve my memory?",
        "Practice spaced repetition, get adequate sleep, exercise regularly, stay mentally active, use mnemonic devices, and reduce stress.",
    ),
    (
        "How do I learn a new language?",
        "Practice daily, immerse yourself through media and conversation, use spaced repetition apps, focus on common vocabulary first, and don't be afraid to make mistakes.",
    ),
    (
        "How do I give a good presentation?",
        "Know your material, practice multiple times, start with a hook, use visual aids sparingly, maintain eye contact, speak clearly, and keep it concise.",
    ),
    (
        "How do I take better photos?",
        "Use natural lighting, follow the rule of thirds, keep the background simple, focus on the subject, experiment with angles, and keep the lens clean.",
    ),
    (
        "How do I manage my time better?",
        "Prioritize tasks, use a calendar or planner, break large tasks into smaller ones, eliminate distractions, set deadlines, and learn to say no.",
    ),
    (
        "How do I negotiate a raise?",
        "Research market salary rates, document your accomplishments, choose the right timing, practice your pitch, present data to support your case, and be confident but flexible.",
    ),
    (
        "How do I read faster?",
        "Reduce subvocalization, use a pointer to guide your eyes, expand your peripheral vision, preview the text before reading, and practice regularly.",
    ),
    (
        "How do I wake up earlier?",
        "Set a consistent sleep schedule, put your alarm across the room, expose yourself to light immediately, avoid screens before bed, and go to bed earlier.",
    ),
    (
        "How do I start exercising?",
        "Start small with 10-15 minutes daily, choose activities you enjoy, set realistic goals, gradually increase intensity, and make it a habit by exercising at the same time each day.",
    ),
    (
        "How do I organize my desk?",
        "Remove everything, clean the surface, sort items into keep/discard/archive, use organizers for supplies, maintain a clean workspace daily.",
    ),
    (
        "How do I deal with procrastination?",
        "Break tasks into small steps, start with the easiest part, set a timer for 5 minutes, remove distractions, reward yourself for progress, and forgive yourself for delays.",
    ),
    (
        "How do I make friends?",
        "Show genuine interest in others, be a good listener, join groups or activities you enjoy, follow up with people, be reliable, and be yourself.",
    ),
]

COMPARISONS = [
    (
        "What is the difference between a lake and an ocean?",
        "A lake is an inland body of fresh water, while an ocean is a vast body of salt water covering most of Earth's surface. Oceans are much larger and deeper.",
    ),
    (
        "What is the difference between weather and climate?",
        "Weather refers to short-term atmospheric conditions (today's rain), while climate describes long-term average weather patterns in an area (tropical climate).",
    ),
    (
        "What is the difference between a virus and a bacteria?",
        "Bacteria are living single-celled organisms that can reproduce on their own. Viruses are not alive by themselves and need a host cell to replicate. Antibiotics work against bacteria but not viruses.",
    ),
    (
        "What is the difference between mass and weight?",
        "Mass is the amount of matter in an object and stays constant. Weight is the force of gravity on that mass and changes depending on gravitational strength.",
    ),
    (
        "What is the difference between speed and velocity?",
        "Speed is how fast an object moves (scalar). Velocity is speed with a direction (vector). A car going 60 mph north has both speed and velocity.",
    ),
    (
        "What is the difference between a meteor and a meteorite?",
        "A meteor is a streak of light in the sky when a space rock burns up in Earth's atmosphere. A meteorite is a space rock that survives and lands on Earth.",
    ),
    (
        "What is the difference between AC and DC?",
        "AC (alternating current) changes direction periodically and is used in homes. DC (direct current) flows in one direction and is used in batteries and electronics.",
    ),
    (
        "What is the difference between a fruit and a vegetable?",
        "Botanically, a fruit develops from a flower and contains seeds. A vegetable is any other edible part of a plant like roots, stems, or leaves.",
    ),
    (
        "What is the difference between an alligator and a crocodile?",
        "Alligators have broader, U-shaped snouts, while crocodiles have longer, V-shaped snouts. Crocodiles can tolerate saltwater, whereas alligators prefer freshwater.",
    ),
    (
        "What is the difference between a democracy and a republic?",
        "A democracy is a system where citizens vote directly on laws. A republic is where citizens elect representatives to make laws on their behalf. Most modern democracies are republics.",
    ),
    (
        "What is the difference between empathy and sympathy?",
        "Empathy is understanding and sharing another's feelings (feeling with them). Sympathy is feeling concern or pity for another's situation (feeling for them).",
    ),
    (
        "What is the difference between a hurricane and a tornado?",
        "Hurricanes are large, long-lasting storms that form over warm ocean water. Tornadoes are smaller, shorter-lived rotating columns of air that form over land.",
    ),
    (
        "What is the difference between heat and temperature?",
        "Heat is the total energy of molecular movement. Temperature is the average energy of molecular movement. A bathtub of warm water has more heat than a cup of boiling water.",
    ),
    (
        "What is the difference between RAM and ROM?",
        "RAM is volatile memory for active data that clears when powered off. ROM is non-volatile memory that retains data permanently and stores firmware.",
    ),
    (
        "What is the difference between software and hardware?",
        "Hardware is the physical components of a computer (CPU, RAM, screen). Software is the programs and instructions that run on the hardware.",
    ),
    (
        "Is a tomato a fruit or a vegetable?",
        "Botanically, a tomato is a fruit because it develops from a flower and contains seeds. In cooking, it is treated as a vegetable.",
    ),
    (
        "What is the difference between a star and a planet?",
        "Stars produce their own light through nuclear fusion. Planets do not produce their own light and orbit around stars. Stars are much larger and hotter.",
    ),
    (
        "What is the difference between a poem and a song?",
        "A poem is a written literary form focused on language, rhythm, and imagery. A song combines words (lyrics) with melody and musical composition.",
    ),
    (
        "What is the difference between a fact and an opinion?",
        "A fact is a statement that can be proven true or false with evidence. An opinion is a personal belief or judgment that cannot be proven objectively.",
    ),
    (
        "What is the difference between longitude and latitude?",
        "Latitude measures distance north or south of the equator (horizontal lines). Longitude measures distance east or west of the Prime Meridian (vertical lines).",
    ),
]

PHIL_QA = [
    (
        "What is the meaning of life?",
        "The meaning of life is one of philosophy's deepest questions. Different perspectives include pursuing happiness, helping others, seeking knowledge, creating meaning through your actions, or following religious or spiritual beliefs.",
    ),
    (
        "What is philosophy?",
        "Philosophy is the study of fundamental questions about existence, knowledge, values, reason, and mind. The word comes from Greek meaning 'love of wisdom.'",
    ),
    (
        "What is ethics?",
        "Ethics is the branch of philosophy dealing with what is morally right and wrong. It examines principles of conduct and how we should treat others.",
    ),
    (
        "What is logic?",
        "Logic is the study of valid reasoning. It provides rules for distinguishing correct arguments from incorrect ones and is fundamental to mathematics, science, and philosophy.",
    ),
    (
        "What is truth?",
        "Truth is the correspondence between a statement and reality. Philosophers debate whether truth is objective, subjective, or determined by consensus.",
    ),
    (
        "What is consciousness?",
        "Consciousness is the state of being aware of and able to think about one's own existence, thoughts, and surroundings. It remains one of the biggest mysteries in science and philosophy.",
    ),
    (
        "What is free will?",
        "Free will is the ability to make choices that are not predetermined. Philosophers debate whether our decisions are truly free or determined by prior causes.",
    ),
    (
        "What is justice?",
        "Justice is the concept of fairness and the administration of what is morally right. It involves equitable treatment and proportional consequences.",
    ),
    (
        "Does the universe have a purpose?",
        "This is debated. Some believe the universe has a designed purpose, while others view it as lacking inherent meaning, leaving individuals to create their own purpose.",
    ),
    (
        "What is happiness?",
        "Happiness can be defined as a state of well-being, contentment, and fulfillment. Philosophers distinguish between momentary pleasure and lasting flourishing.",
    ),
    (
        "What is wisdom?",
        "Wisdom is the ability to apply knowledge and experience with good judgment. It involves understanding, discernment, and making sound decisions.",
    ),
    (
        "What is the trolley problem?",
        "The trolley problem is an ethical thought experiment: a runaway trolley will kill five people unless you divert it to a track where it will kill one person. It explores utilitarian vs. deontological ethics.",
    ),
    (
        "What is existentialism?",
        "Existentialism is a philosophical movement emphasizing individual existence, freedom, and choice. Key thinkers include Sartre, Kierkegaard, and Camus.",
    ),
    (
        "What is morality?",
        "Morality refers to principles distinguishing right from wrong behavior. It can be based on cultural norms, religious teachings, or philosophical reasoning.",
    ),
    (
        "What is a paradox?",
        "A paradox is a seemingly contradictory statement that may reveal a deeper truth. The classic example: 'This statement is false.' If true, it is false; if false, it is true.",
    ),
    (
        "What is critical thinking?",
        "Critical thinking is the disciplined process of analyzing information objectively, questioning assumptions, evaluating evidence, and forming reasoned judgments.",
    ),
    (
        "Why do we dream?",
        "The exact purpose of dreams is debated. Theories include memory consolidation, emotional processing, problem solving, brain maintenance, and threat simulation.",
    ),
    (
        "What is the nature of reality?",
        "Philosophers debate whether reality exists independently of our perception (realism) or is constructed by our minds (idealism). Science strives to describe reality through observation and testing.",
    ),
    (
        "Can machines think?",
        "This is a central question in AI philosophy. Alan Turing proposed the Turing Test, but whether machines truly 'think' or just simulate thinking remains debated.",
    ),
    (
        "What is the Golden Rule?",
        "The Golden Rule is the ethical principle of treating others as you would want to be treated. Versions of it appear in virtually every major religion and philosophical tradition.",
    ),
]

ARTS_QA = [
    (
        "Who painted the Mona Lisa?",
        "The Mona Lisa was painted by Leonardo da Vinci, an Italian Renaissance artist, around 1503-1519. It is on display at the Louvre Museum in Paris.",
    ),
    (
        "Who wrote Romeo and Juliet?",
        "Romeo and Juliet was written by William Shakespeare, the famous English playwright, around 1594-1596.",
    ),
    (
        "What is a symphony?",
        "A symphony is an extended musical composition for a full orchestra, typically in four movements. Famous composers of symphonies include Beethoven, Mozart, and Brahms.",
    ),
    (
        "What is abstract art?",
        "Abstract art does not attempt to represent reality accurately. Instead, it uses shapes, colors, and forms to convey emotions or ideas.",
    ),
    (
        "What is an opera?",
        "An opera is a dramatic work combining text (libretto) with a musical score. Singers perform the roles accompanied by an orchestra.",
    ),
    (
        "Who was Beethoven?",
        "Ludwig van Beethoven was a German composer and pianist who bridged the Classical and Romantic periods. He continued composing masterpieces even after becoming deaf.",
    ),
    (
        "Who was Mozart?",
        "Wolfgang Amadeus Mozart was an Austrian composer who began composing at age five and wrote over 600 works including symphonies, operas, and concertos.",
    ),
    (
        "What is jazz?",
        "Jazz is a music genre that originated in the African American communities of New Orleans in the early 20th century. It features improvisation, swing rhythms, and complex harmonies.",
    ),
    (
        "What is a novel?",
        "A novel is a long fictional narrative written in prose. It typically features characters, a plot, setting, theme, and explores the human experience.",
    ),
    (
        "What is poetry?",
        "Poetry is a form of literature that uses aesthetic and rhythmic qualities of language to express meaning. It often employs meter, rhyme, and imagery.",
    ),
    (
        "What is a haiku?",
        "A haiku is a traditional Japanese poem with three lines of 5, 7, and 5 syllables. It usually captures a moment in nature.",
    ),
    (
        "What is the Renaissance in art?",
        "The Renaissance was a period (14th-17th century) of renewed interest in classical art, featuring perspective, realism, and humanism. Key artists include da Vinci, Michelangelo, and Raphael.",
    ),
    (
        "Who painted the Starry Night?",
        "Starry Night was painted by Vincent van Gogh in June 1889 while he was staying at an asylum. It depicts a swirling night sky over a village.",
    ),
    (
        "What is a documentary?",
        "A documentary is a non-fiction film or television program that provides a factual account of a subject, often using real footage, interviews, and narration.",
    ),
    (
        "What is ballet?",
        "Ballet is a highly formalized dance form that originated in Renaissance Italy and developed in France and Russia. It features precise movements and pointe technique.",
    ),
    (
        "Who was Shakespeare?",
        "William Shakespeare was an English playwright and poet (1564-1616), widely regarded as the greatest writer in the English language. He wrote 37 plays and 154 sonnets.",
    ),
    (
        "What is a sonnet?",
        "A sonnet is a 14-line poem with a specific rhyme scheme. The two main forms are the Italian (Petrarchan) sonnet and the English (Shakespearean) sonnet.",
    ),
    (
        "What is impressionism?",
        "Impressionism is an art movement that began in 1860s France. Artists like Monet, Renoir, and Degas used visible brushstrokes and light to capture moments.",
    ),
    (
        "What is a mural?",
        "A mural is a large artwork painted or applied directly on a wall, ceiling, or other large surface. Famous examples include Michelangelo's Sistine Chapel ceiling.",
    ),
    (
        "What is classical music?",
        "Classical music is a broad term for Western orchestral tradition spanning from roughly 1750 to 1820, though informally it refers to all Western art music from medieval to modern periods.",
    ),
]

SPORTS_QA = [
    (
        "What is soccer?",
        "Soccer (association football) is the world's most popular sport. Two teams of 11 players try to score by kicking a ball into the opposing goal.",
    ),
    (
        "What is basketball?",
        "Basketball is a sport where two teams of five players score points by shooting a ball through a hoop. It was invented by James Naismith in 1891.",
    ),
    (
        "What is tennis?",
        "Tennis is a racket sport played individually or in pairs. Players hit a ball over a net, scoring points when opponents cannot return the ball.",
    ),
    (
        "What are the Olympic Games?",
        "The Olympic Games are an international multi-sport event held every four years, featuring athletes from over 200 countries competing in dozens of sports.",
    ),
    (
        "What is a marathon?",
        "A marathon is a long-distance running race covering 42.195 kilometers (26.2 miles). It originated from the legend of a Greek messenger running from Marathon to Athens.",
    ),
    (
        "What is baseball?",
        "Baseball is a bat-and-ball sport where two teams of nine players take turns batting and fielding. The objective is to score runs by hitting the ball and running bases.",
    ),
    (
        "What is cricket?",
        "Cricket is a bat-and-ball sport played between two teams of 11 players.  One team bats trying to score runs while the other bowls and fields.",
    ),
    (
        "What is golf?",
        "Golf is a sport where players use clubs to hit balls into holes on a course, aiming to complete each hole in as few strokes as possible.",
    ),
    (
        "What is swimming?",
        "Swimming is an individual or team sport involving propelling oneself through water using arms and legs. Competitive events include freestyle, backstroke, breaststroke, and butterfly.",
    ),
    (
        "How big is a football field?",
        "An American football field is 100 yards (91.4 meters) long between the end zones and 53.3 yards (48.8 meters) wide.",
    ),
    (
        "What is the World Cup?",
        "The FIFA World Cup is an international soccer tournament held every four years. It is the most watched sporting event in the world.",
    ),
    (
        "What is the Super Bowl?",
        "The Super Bowl is the annual championship game of the National Football League (NFL) in the United States. It is one of the most-watched television events in America.",
    ),
    (
        "What is boxing?",
        "Boxing is a combat sport where two opponents throw punches at each other in a ring. Matches are won by knockout, decision, or stoppage.",
    ),
    (
        "What is gymnastics?",
        "Gymnastics is a sport involving exercises requiring balance, strength, flexibility, and coordination. Events include floor, vault, bars, and beam.",
    ),
    (
        "Who holds the 100m world record?",
        "As of my knowledge, Usain Bolt holds the 100m world record at 9.58 seconds, set in Berlin in 2009.",
    ),
    (
        "What is a hat trick?",
        "A hat trick originally referred to three wickets in cricket. In soccer and hockey, it means one player scoring three goals in a single game.",
    ),
    (
        "What is MMA?",
        "MMA (Mixed Martial Arts) is a full-contact combat sport that combines techniques from boxing, wrestling, jiu-jitsu, muay thai, and other martial arts.",
    ),
    (
        "What is volleyball?",
        "Volleyball is a team sport where two teams of six players are separated by a net and try to score by grounding the ball on the other team's court.",
    ),
    (
        "What is rugby?",
        "Rugby is a contact team sport where players carry, pass, and kick a ball to score tries by grounding it in the opposing team's goal area.",
    ),
    (
        "What is ice hockey?",
        "Ice hockey is a fast-paced sport played on ice where two teams of six use sticks to shoot a puck into the opposing team's goal.",
    ),
]

ECON_QA = [
    (
        "What is inflation?",
        "Inflation is the rate at which the general price level of goods and services rises over time, reducing purchasing power.",
    ),
    (
        "What is GDP?",
        "GDP (Gross Domestic Product) is the total monetary value of all goods and services produced within a country's borders in a specific time period.",
    ),
    (
        "What is supply and demand?",
        "Supply and demand is an economic model describing how prices are set. When demand increases or supply decreases, prices rise. When demand decreases or supply increases, prices fall.",
    ),
    (
        "What is a stock?",
        "A stock represents a share of ownership in a company. When you buy stock, you become a partial owner and may receive dividends and capital gains.",
    ),
    (
        "What is a bond?",
        "A bond is a debt instrument where an investor lends money to an entity (government or corporation) for a fixed period at a set interest rate.",
    ),
    (
        "What is a recession?",
        "A recession is a significant decline in economic activity lasting more than a few months, typically visible in GDP, employment, income, and industrial production.",
    ),
    (
        "What is interest rate?",
        "An interest rate is the percentage charged by a lender to a borrower for the use of money, or the percentage paid by a bank on deposits.",
    ),
    (
        "What is a budget?",
        "A budget is a financial plan that estimates income and expenses over a specific period, helping individuals or organizations manage money effectively.",
    ),
    (
        "What is cryptocurrency?",
        "Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates on decentralized networks, typically using blockchain technology.",
    ),
    (
        "What is an entrepreneur?",
        "An entrepreneur is a person who starts and runs a business, taking on financial risks in hope of profit while often innovating products or services.",
    ),
    (
        "What is a monopoly?",
        "A monopoly exists when one company or entity is the sole provider of a product or service in a market, with no significant competition.",
    ),
    (
        "What is capitalism?",
        "Capitalism is an economic system where private individuals own and control the means of production, and prices are determined by market competition.",
    ),
    (
        "What is socialism?",
        "Socialism is an economic system where the means of production are owned collectively or by the government, aiming for more equitable distribution of resources.",
    ),
    (
        "What is a patent?",
        "A patent is a legal right granted to an inventor that prevents others from making, using, or selling their invention for a set period, typically 20 years.",
    ),
    (
        "What is a startup?",
        "A startup is a newly established business, often technology-based, that aims to grow rapidly by developing an innovative product or service.",
    ),
    (
        "What is marketing?",
        "Marketing is the process of promoting, selling, and distributing products or services. It involves market research, advertising, and building customer relationships.",
    ),
    (
        "What is a tax?",
        "A tax is a mandatory financial charge imposed by a government on individuals or businesses to fund public services and infrastructure.",
    ),
    (
        "What is compound interest?",
        "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods, causing exponential growth over time.",
    ),
    (
        "What is a trade deficit?",
        "A trade deficit occurs when a country imports more goods and services than it exports, meaning it spends more on foreign products than it earns from selling abroad.",
    ),
    (
        "What is the stock market?",
        "The stock market is a system where shares of publicly traded companies are bought and sold, allowing companies to raise capital and investors to own portions of businesses.",
    ),
]

CAP_QA = [
    (
        "Can you search the internet?",
        "No, I cannot search the internet. I run completely offline. All my responses come from my local language model.",
    ),
    (
        "Can you go online?",
        "No, I run entirely offline. I do not have any internet connectivity.",
    ),
    (
        "Do you need the internet?",
        "No, I do not need the internet at all. Everything I do runs locally.",
    ),
    (
        "Are you connected to the internet?",
        "No, I am not connected to the internet. I am a fully offline assistant.",
    ),
    (
        "Can you send a message?",
        "No, I cannot send messages because I run entirely offline with no internet connection.",
    ),
    (
        "Can you make a phone call?",
        "No, I cannot make phone calls. I am an offline assistant.",
    ),
    (
        "Can you browse the web?",
        "No, I cannot browse the web because I run completely offline.",
    ),
    (
        "Can you set a timer?",
        "I am unable to set timers as that capability is not currently available.",
    ),
    (
        "Can you set an alarm?",
        "I am unable to set alarms as that capability is not currently available.",
    ),
    ("Can you play music?", "I am designed for conversation, not music playback."),
    (
        "Can you control my lights?",
        "I cannot control smart home devices because I run fully offline with no home automation connectivity.",
    ),
    (
        "Is my data private?",
        "Yes, your data is completely private. I run entirely offline. Nothing you say is sent anywhere.",
    ),
    (
        "Do you send my data anywhere?",
        "No, I do not send any data anywhere. I run fully offline. All processing happens locally.",
    ),
    (
        "Why do you work offline?",
        "I work offline by design. This means I function without internet and keep your conversations completely private.",
    ),
    (
        "What can you do?",
        "I can answer questions, have conversations, explain concepts, share fun facts, help with math, tell jokes, and discuss many topics. I work entirely offline.",
    ),
    (
        "What are you capable of?",
        "I can have conversations, answer questions, share fun facts, help with general knowledge, and provide information across many topics.",
    ),
    (
        "How can you help me?",
        "I can answer questions, chat with you, explain concepts, tell jokes, share fun facts, and provide information on many topics.",
    ),
    (
        "What are your limitations?",
        "I cannot access the internet, check live weather, tell the current time, control smart devices, or perform complex reasoning. I work best for conversational Q&A.",
    ),
    (
        "What can't you do?",
        "I cannot access real-time information, browse the web, send messages, make calls, or interact with physical devices. My knowledge comes from my training data.",
    ),
    (
        "How do you work?",
        "I use a local language model to understand your input and generate helpful responses. Everything runs offline on your machine.",
    ),
    (
        "How do you generate responses?",
        "I use a fine-tuned language model running locally. I predict the most likely next words based on your input and my training.",
    ),
    (
        "What language model do you use?",
        "I use a fine-tuned language model optimized to run efficiently on local hardware.",
    ),
    (
        "Why should I use you instead of ChatGPT?",
        "I run fully offline. Your conversations stay private. I have no subscription costs and work anywhere without an internet connection.",
    ),
    (
        "Can you speak other languages?",
        "I was trained primarily in English. My ability in other languages is limited since I run a small local model.",
    ),
    (
        "Can you remember our conversation?",
        "I can keep track of our current conversation, but I do not remember between sessions.",
    ),
]

ENV_QA = [
    (
        "What is global warming?",
        "Global warming is the long-term increase in Earth's average surface temperature, primarily caused by greenhouse gas emissions from burning fossil fuels.",
    ),
    (
        "What are greenhouse gases?",
        "Greenhouse gases trap heat in Earth's atmosphere. Major ones include carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and water vapor.",
    ),
    (
        "What is the greenhouse effect?",
        "The greenhouse effect is when gases in Earth's atmosphere trap heat from the Sun, keeping the planet warm enough to support life. Human activity has intensified this effect.",
    ),
    (
        "What is renewable energy?",
        "Renewable energy comes from sources that are naturally replenished, like solar, wind, hydroelectric, geothermal, and biomass energy.",
    ),
    (
        "What is solar energy?",
        "Solar energy is power generated from sunlight, either through photovoltaic panels that convert light to electricity or solar thermal systems that use heat.",
    ),
    (
        "What is wind energy?",
        "Wind energy uses wind turbines to convert the kinetic energy of wind into electrical energy. It is one of the fastest-growing renewable energy sources.",
    ),
    (
        "What is deforestation?",
        "Deforestation is the clearing of forests for agriculture, logging, or development. It contributes to climate change, habitat loss, and biodiversity decline.",
    ),
    (
        "What is biodiversity?",
        "Biodiversity is the variety of life on Earth, including the diversity of species, genes, and ecosystems. It is essential to ecosystem health and resilience.",
    ),
    (
        "Why are bees important?",
        "Bees are critical pollinators for many food crops and wild plants. About one-third of the food we eat depends on bee pollination.",
    ),
    (
        "What is recycling?",
        "Recycling is the process of converting waste materials into new materials and objects. It reduces the need for raw materials and decreases waste in landfills.",
    ),
    (
        "What is an endangered species?",
        "An endangered species is one that faces a very high risk of extinction in the wild. Examples include giant pandas, tigers, and blue whales.",
    ),
    (
        "What is the carbon footprint?",
        "A carbon footprint is the total amount of greenhouse gases generated by a person, organization, or activity, usually measured in tons of CO2 equivalent.",
    ),
    (
        "What is ocean acidification?",
        "Ocean acidification is the decrease in ocean pH caused by absorbing excess CO2 from the atmosphere. It threatens marine life, especially shell-forming organisms.",
    ),
    (
        "What is air pollution?",
        "Air pollution is the presence of harmful substances in the atmosphere, including particulate matter, ozone, nitrogen dioxide, and sulfur dioxide, affecting health and environment.",
    ),
    (
        "What is a rainforest?",
        "A rainforest is a dense forest with high annual rainfall. Tropical rainforests near the equator are the most biodiverse ecosystems on Earth.",
    ),
    (
        "Why are coral reefs important?",
        "Coral reefs support about 25% of all marine species, protect coastlines from storms, provide food and income for millions of people, and support tourism.",
    ),
    (
        "What is sustainability?",
        "Sustainability means meeting present needs without compromising future generations' ability to meet their needs, balancing environmental, economic, and social factors.",
    ),
    (
        "What is composting?",
        "Composting is the natural process of recycling organic matter like food scraps and yard waste into nutrient-rich soil amendment for gardens.",
    ),
    (
        "What causes acid rain?",
        "Acid rain is caused by sulfur dioxide and nitrogen oxide emissions from burning fossil fuels. These gases react with water in the atmosphere to form acidic rainfall.",
    ),
    (
        "What is nuclear energy?",
        "Nuclear energy is produced by splitting atoms (fission) in a reactor. It generates large amounts of electricity with very low carbon emissions but produces radioactive waste.",
    ),
]


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================


def _add_originals(base_examples, seen):
    """
    Adds original examples to the result list while tracking seen keys.

    Args:
        base_examples (list): The list of base examples to process.
        seen (set): A set tracking already processed user-assistant pairs.

    Returns:
        list: A list containing the newly added original examples.
    """
    result = []
    for ex in base_examples:
        key = (ex["user"], ex["assistant"])
        if key not in seen:
            seen.add(key)
            result.append(ex)
    return result


def _get_q_variation(q):
    """
    Generates a variation of the user question.

    Args:
        q (str): The original user question.

    Returns:
        str: A formatted question variation.
    """
    ql = q[0].lower() + q[1:] if q[0].isupper() else q
    return random.choice(Q_TEMPLATES).format(q=q, ql=ql)


def _get_a_variation(a, skip_a):
    """
    Generates a variation of the assistant response.

    Args:
        a (str): The original assistant response.
        skip_a (bool): Whether to skip variation generation.

    Returns:
        str: The varied or original assistant response.
    """
    if skip_a:
        return a
    return random.choice(A_OPENERS).format(a=a)


def _process_single_variation(result, seen, ex, skip_a):
    """
    Processes and conditionally adds a single generated variation.

    Args:
        result (list): The list accumulating unique examples.
        seen (set): A set of already processed user-assistant pairs.
        ex (dict): The base example to vary.
        skip_a (bool): Whether to skip variation of the assistant text.
    """
    q_var = _get_q_variation(ex["user"])
    a_var = _get_a_variation(ex["assistant"], skip_a)
    key = (q_var, a_var)
    if key not in seen:
        seen.add(key)
        result.append({"user": q_var, "assistant": a_var})


def _generate_variations(base, result, seen, n, skip_a):
    """
    Generates variations until the target count n is reached.

    Args:
        base (list): The list of base examples to sample from.
        result (list): The list accumulating unique examples.
        seen (set): A set of already processed user-assistant pairs.
        n (int): The target number of total examples.
        skip_a (bool): Whether to skip variation of the assistant text.

    Returns:
        list: The updated list of diversified examples.
    """
    attempts = 0
    while len(result) < n and attempts < n * 20:
        attempts += 1
        ex = random.choice(base)
        _process_single_variation(result, seen, ex, skip_a)
    return result


def diversify_examples(examples, n, skip_a=False):
    """
    Expands a list of base examples to n unique pairs.

    Args:
        examples (list): The base list of examples to expand.
        n (int): The target number of total examples.
        skip_a (bool): Whether to skip variation of the assistant text.

    Returns:
        list: A list of length n containing unique varied examples.
    """
    seen = set()
    result = _add_originals(examples, seen)
    result = _generate_variations(examples, result, seen, n, skip_a)
    random.shuffle(result)
    return result[:n]


def check_identity(d):
    """
    Checks if a given dictionary example contains identity keywords.

    Args:
        d (dict): The dictionary containing user and assistant keys.

    Returns:
        bool: True if identity keywords are found in the assistant text, False otherwise.
    """
    text = d["assistant"].lower()
    return "embedie" in text or "kevin thomas" in text


# ===========================================================================
# SCRIPT-LEVEL LOGIC
# ===========================================================================

# Initialize the global accumulator
print("Generating training data...")
all_examples = []

# --- IDENTITY LOGIC ---
identity_base = list(IDENTITY)
identity_prefixes = [
    "Hey, ",
    "Hi, ",
    "So, ",
    "Just wondering, ",
    "Quick question: ",
    "I want to know, ",
    "Please tell me, ",
    "Can you tell me, ",
]
identity_extras = []
for ex in IDENTITY:
    for prefix in identity_prefixes:
        if len(identity_base) + len(identity_extras) >= 250:
            break
        first_char = ex["user"][0]
        modified_user = (
            prefix.rstrip(", ") + " " + first_char.lower() + ex["user"][1:]
            if first_char.isupper()
            else prefix + ex["user"]
        )
        identity_extras.append({"user": modified_user, "assistant": ex["assistant"]})
random.shuffle(identity_extras)
all_examples.extend(identity_base + identity_extras[: 250 - len(identity_base)])
print(f"  Identity: 250 examples")

# --- GREETINGS LOGIC ---
greetings_base = []
for text in GREETINGS_IN:
    greetings_base.append({"user": text, "assistant": random.choice(GREETINGS_OUT)})
for text in FAREWELL_IN:
    greetings_base.append({"user": text, "assistant": random.choice(FAREWELL_OUT)})
for text in THANKS_IN:
    greetings_base.append({"user": text, "assistant": random.choice(THANKS_OUT)})
greetings_base.extend([{"user": q, "assistant": a} for q, a in SMALLTALK])
all_examples.extend(diversify_examples(greetings_base, 400))
print(f"  Greetings & Small Talk: 400 examples")

# --- SCIENCE LOGIC ---
science_base = [{"user": q, "assistant": a} for q, a in SCIENCE_QA]
sci_alt_phrasing = [
    "Can you explain {}?",
    "What do you know about {}?",
    "Tell me about {}.",
]
for q, a in SCIENCE_QA:
    topic = (
        q.replace("What is ", "")
        .replace("What are ", "")
        .replace("What causes ", "")
        .rstrip("?")
        .lower()
    )
    for tmpl in sci_alt_phrasing:
        science_base.append({"user": tmpl.format(topic), "assistant": a})
all_examples.extend(diversify_examples(science_base, 1200))
print(f"  Science: 1200 examples")

# --- MATH LOGIC ---
math_base = []
for _ in range(200):
    a, b = random.randint(1, 999), random.randint(1, 999)
    op = random.choice(["+", "-", "*"])
    if op == "+":
        math_base.append(
            {"user": f"What is {a} + {b}?", "assistant": f"{a} + {b} = {a+b}."}
        )
    elif op == "-":
        math_base.append(
            {"user": f"What is {a} - {b}?", "assistant": f"{a} - {b} = {a-b}."}
        )
    else:
        a, b = random.randint(2, 50), random.randint(2, 50)
        math_base.append(
            {"user": f"What is {a} times {b}?", "assistant": f"{a} times {b} is {a*b}."}
        )
for _ in range(50):
    b = random.randint(2, 20)
    result = random.randint(1, 50)
    a = b * result
    math_base.append(
        {
            "user": f"What is {a} divided by {b}?",
            "assistant": f"{a} divided by {b} is {result}.",
        }
    )
for i in range(2, 26):
    math_base.append(
        {"user": f"What is {i} squared?", "assistant": f"{i} squared is {i**2}."}
    )
for i in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
    math_base.append(
        {
            "user": f"What is the square root of {i}?",
            "assistant": f"The square root of {i} is {int(i**0.5)}.",
        }
    )
for i in range(1, 21):
    math_base.append(
        {
            "user": f"What is 2 to the power of {i}?",
            "assistant": f"2 to the power of {i} is {2**i}.",
        }
    )
for q, a in MATH_CONCEPTS:
    math_base.append({"user": q, "assistant": a})
for _ in range(50):
    pct = random.choice([10, 15, 20, 25, 30, 50, 75])
    val = random.choice([100, 200, 500, 1000, 50, 80, 150, 250, 400])
    result = pct * val / 100
    res_str = f"{int(result)}" if result == int(result) else f"{result}"
    math_base.append(
        {
            "user": f"What is {pct}% of {val}?",
            "assistant": f"{pct}% of {val} is {res_str}.",
        }
    )
all_examples.extend(diversify_examples(math_base, 800))
print(f"  Math: 800 examples")

# --- GEOGRAPHY LOGIC ---
geo_base = []
for country, capital in CAPITALS:
    geo_base.append(
        {
            "user": f"What is the capital of {country}?",
            "assistant": f"The capital of {country} is {capital}.",
        }
    )
    geo_base.append(
        {
            "user": f"What city is the capital of {country}?",
            "assistant": f"{capital} is the capital of {country}.",
        }
    )
for q, a in CONTINENTS + OCEANS + GEO_FACTS:
    geo_base.append({"user": q, "assistant": a})
all_examples.extend(diversify_examples(geo_base, 800))
print(f"  Geography: 800 examples")

# --- HISTORY LOGIC ---
history_base = [{"user": q, "assistant": a} for q, a in HISTORY_QA]
hist_prefixes = [
    "Tell me about ",
    "Can you explain ",
    "What do you know about ",
    "I'd like to learn about ",
]
for q, a in HISTORY_QA:
    topic = (
        q.replace("What was ", "")
        .replace("Who was ", "")
        .replace("When did ", "")
        .replace("When was ", "")
        .rstrip("?")
    )
    for p in hist_prefixes[:2]:
        history_base.append({"user": f"{p}{topic.lower()}.", "assistant": a})
all_examples.extend(diversify_examples(history_base, 800))
print(f"  History: 800 examples")

# --- CODING LOGIC ---
coding_base = [{"user": q, "assistant": a} for q, a in CODING_QA]
for q, a in CODING_QA[:20]:
    topic = q.replace("What is ", "").rstrip("?")
    coding_base.append({"user": f"Explain {topic} to me.", "assistant": a})
    coding_base.append(
        {"user": f"How would you describe {topic.lower()}?", "assistant": a}
    )
all_examples.extend(diversify_examples(coding_base, 800))
print(f"  Coding & Technology: 800 examples")

# --- FUN FACTS LOGIC ---
facts_base = [{"user": q, "assistant": a} for q, a in FACTS_QA]
seen_facts = set((e["user"], e["assistant"]) for e in facts_base)
fact_attempts = 0
while len(facts_base) < 500 and fact_attempts < 10000:
    fact_attempts += 1
    _, ans = random.choice(FACTS_QA)
    prompt = random.choice(FACT_PROMPTS)
    if (prompt, ans) not in seen_facts:
        seen_facts.add((prompt, ans))
        facts_base.append({"user": prompt, "assistant": ans})
random.shuffle(facts_base)
all_examples.extend(facts_base[:500])
print(f"  Fun Facts: 500 examples")

# --- JOKES LOGIC ---
jokes_base = [{"user": q, "assistant": a} for q, a in JOKES_QA]
seen_jokes = set((e["user"], e["assistant"]) for e in jokes_base)
joke_attempts = 0
while len(jokes_base) < 400 and joke_attempts < 8000:
    joke_attempts += 1
    _, ans = random.choice(JOKES_QA)
    prompt = random.choice(JOKE_PROMPTS)
    if (prompt, ans) not in seen_jokes:
        seen_jokes.add((prompt, ans))
        jokes_base.append({"user": prompt, "assistant": ans})
random.shuffle(jokes_base)
all_examples.extend(jokes_base[:400])
print(f"  Jokes: 400 examples")

# --- ANIMALS LOGIC ---
animals_base = [{"user": q, "assistant": a} for q, a in ANIMALS_QA]
for q, a in ANIMALS_QA:
    topic = q.split("?")[0].split("do ")[-1].split("is ")[-1].lower().strip()
    animals_base.append({"user": f"Tell me about {topic}.", "assistant": a})
all_examples.extend(diversify_examples(animals_base, 500))
print(f"  Animals & Nature: 500 examples")

# --- HEALTH LOGIC ---
health_base = [{"user": q, "assistant": a} for q, a in HEALTH_QA]
for q, a in HEALTH_QA:
    topic = (
        q.replace("What is ", "")
        .replace("What are ", "")
        .replace("How ", "how ")
        .replace("Why ", "why ")
        .rstrip("?")
    )
    health_base.append({"user": f"Can you explain {topic}?", "assistant": a})
    health_base.append({"user": f"I'd like to know: {q.lower()}", "assistant": a})
all_examples.extend(diversify_examples(health_base, 400))
print(f"  Health & Fitness: 400 examples")

# --- FOOD LOGIC ---
food_base = [{"user": q, "assistant": a} for q, a in FOOD_QA]
for q, a in FOOD_QA:
    food_base.append({"user": f"Can you tell me {q.lower()}", "assistant": a})
    food_base.append({"user": f"I want to know: {q.lower()}", "assistant": a})
all_examples.extend(diversify_examples(food_base, 400))
print(f"  Food & Cooking: 400 examples")

# --- LANGUAGE LOGIC ---
lang_base = [{"user": q, "assistant": a} for q, a in LANG_QA]
for q, a in LANG_QA[:15]:
    topic = q.split("mean?")[0].replace("What does ", "").strip().strip("'").lower()
    lang_base.append({"user": f"Define {topic} for me.", "assistant": a})
all_examples.extend(diversify_examples(lang_base, 400))
print(f"  Language & Vocabulary: 400 examples")

# --- HOW-TO LOGIC ---
howto_base = [{"user": q, "assistant": a} for q, a in HOWTO_QA]
for q, a in HOWTO_QA:
    alt1 = q.replace("How do I ", "What is the best way to ").rstrip("?") + "?"
    alt2 = q.replace("How do I ", "Can you tell me how to ").rstrip("?") + "?"
    howto_base.extend([{"user": alt1, "assistant": a}, {"user": alt2, "assistant": a}])
all_examples.extend(diversify_examples(howto_base, 500))
print(f"  How-To & Practical: 500 examples")

# --- COMPARISONS LOGIC ---
comp_base = [{"user": q, "assistant": a} for q, a in COMPARISONS]
for q, a in COMPARISONS:
    alt1 = q.replace("What is the difference between ", "Compare ").replace("?", ".")
    alt2 = q.replace("What is the difference between", "How are").replace(
        "?", " different?"
    )
    comp_base.extend([{"user": alt1, "assistant": a}, {"user": alt2, "assistant": a}])
all_examples.extend(diversify_examples(comp_base, 400))
print(f"  Comparisons: 400 examples")

# --- PHILOSOPHY LOGIC ---
phil_base = [{"user": q, "assistant": a} for q, a in PHIL_QA]
for q, a in PHIL_QA:
    phil_base.append({"user": f"I've been thinking: {q.lower()}", "assistant": a})
    phil_base.append(
        {
            "user": f"Can you explain {q.replace('What is ', '').rstrip('?').lower()}?",
            "assistant": a,
        }
    )
all_examples.extend(diversify_examples(phil_base, 400))
print(f"  Philosophy: 400 examples")

# --- ARTS LOGIC ---
arts_base = [{"user": q, "assistant": a} for q, a in ARTS_QA]
for q, a in ARTS_QA:
    topic = (
        q.replace("Who was ", "")
        .replace("What is ", "")
        .replace("Who painted ", "")
        .replace("Who wrote ", "")
        .rstrip("?")
        .lower()
    )
    arts_base.append({"user": f"Tell me about {topic}.", "assistant": a})
    arts_base.append({"user": f"I'd like to know about {topic}.", "assistant": a})
all_examples.extend(diversify_examples(arts_base, 400))
print(f"  Arts & Entertainment: 400 examples")

# --- SPORTS LOGIC ---
sports_base = [{"user": q, "assistant": a} for q, a in SPORTS_QA]
for q, a in SPORTS_QA:
    topic = q.replace("What is ", "").replace("What are ", "").rstrip("?").lower()
    sports_base.append({"user": f"Explain {topic} to me.", "assistant": a})
    sports_base.append({"user": f"Can you tell me about {topic}?", "assistant": a})
all_examples.extend(diversify_examples(sports_base, 400))
print(f"  Sports: 400 examples")

# --- ECONOMICS LOGIC ---
econ_base = [{"user": q, "assistant": a} for q, a in ECON_QA]
for q, a in ECON_QA:
    topic = (
        q.replace("What is ", "")
        .replace("What is a ", "")
        .replace("What is an ", "")
        .rstrip("?")
        .lower()
    )
    econ_base.append({"user": f"Explain {topic} to me.", "assistant": a})
    econ_base.append({"user": f"Define {topic}.", "assistant": a})
all_examples.extend(diversify_examples(econ_base, 400))
print(f"  Economics & Business: 400 examples")

# --- CAPABILITIES LOGIC ---
cap_base = [{"user": q, "assistant": a} for q, a in CAP_QA]
for q, a in CAP_QA:
    cap_base.append(
        {
            "user": q.replace("Can you", "Are you able to").replace(
                "can you", "are you able to"
            ),
            "assistant": a,
        }
    )
all_examples.extend(diversify_examples(cap_base, 400))
print(f"  Capabilities & Limitations: 400 examples")

# --- CONVERSIONS LOGIC ---
conv_base = []
for f in range(0, 221, 10):
    conv_base.append(
        {
            "user": f"Convert {f} Fahrenheit to Celsius.",
            "assistant": f"{f}°F is {round((f - 32) * 5 / 9, 1)}°C.",
        }
    )
for c in range(0, 101, 5):
    conv_base.append(
        {
            "user": f"Convert {c} Celsius to Fahrenheit.",
            "assistant": f"{c}°C is {round(c * 9 / 5 + 32, 1)}°F.",
        }
    )
for m in [1, 2, 3, 5, 10, 15, 20, 25, 30, 50, 100, 200, 500, 1000]:
    conv_base.append(
        {
            "user": f"How many kilometers is {m} mile{'s' if m != 1 else ''}?",
            "assistant": f"{m} mile{'s' if m != 1 else ''} is approximately {round(m * 1.60934, 2)} kilometers.",
        }
    )
for km in [1, 2, 5, 10, 20, 42, 50, 100, 200, 500, 1000]:
    conv_base.append(
        {
            "user": f"How many miles is {km} kilometer{'s' if km != 1 else ''}?",
            "assistant": f"{km} kilometer{'s' if km != 1 else ''} is approximately {round(km / 1.60934, 2)} miles.",
        }
    )
for lb in [1, 2, 5, 10, 20, 50, 100, 150, 200]:
    conv_base.append(
        {
            "user": f"Convert {lb} pounds to kilograms.",
            "assistant": f"{lb} pounds is approximately {round(lb * 0.453592, 2)} kilograms.",
        }
    )
for kg in [1, 2, 5, 10, 20, 50, 75, 100]:
    conv_base.append(
        {
            "user": f"Convert {kg} kilograms to pounds.",
            "assistant": f"{kg} kilograms is approximately {round(kg / 0.453592, 2)} pounds.",
        }
    )
for gal in [1, 2, 5, 10, 20]:
    conv_base.append(
        {
            "user": f"How many liters in {gal} gallon{'s' if gal != 1 else ''}?",
            "assistant": f"{gal} gallon{'s' if gal != 1 else ''} is approximately {round(gal * 3.78541, 2)} liters.",
        }
    )
TIME_QA = [
    ("How many seconds in a minute?", "There are 60 seconds in a minute."),
    ("How many minutes in an hour?", "There are 60 minutes in an hour."),
    ("How many hours in a day?", "There are 24 hours in a day."),
    (
        "How many days in a year?",
        "There are 365 days in a regular year and 366 days in a leap year.",
    ),
    ("How many weeks in a year?", "There are 52 weeks in a year."),
    ("How many months in a year?", "There are 12 months in a year."),
    (
        "How many seconds in an hour?",
        "There are 3,600 seconds in an hour (60 minutes times 60 seconds).",
    ),
    (
        "How many seconds in a day?",
        "There are 86,400 seconds in a day (24 hours times 3,600 seconds per hour).",
    ),
    (
        "How many hours in a week?",
        "There are 168 hours in a week (7 days times 24 hours).",
    ),
    (
        "How many minutes in a day?",
        "There are 1,440 minutes in a day (24 hours times 60 minutes).",
    ),
]
DATA_QA = [
    ("How many bytes in a kilobyte?", "There are 1,024 bytes in a kilobyte."),
    ("How many kilobytes in a megabyte?", "There are 1,024 kilobytes in a megabyte."),
    ("How many megabytes in a gigabyte?", "There are 1,024 megabytes in a gigabyte."),
    ("How many gigabytes in a terabyte?", "There are 1,024 gigabytes in a terabyte."),
]
for q, a in TIME_QA + DATA_QA:
    conv_base.append({"user": q, "assistant": a})
all_examples.extend(diversify_examples(conv_base, 500))
print(f"  Conversions & Units: 500 examples")

# --- ENVIRONMENT LOGIC ---
env_base = [{"user": q, "assistant": a} for q, a in ENV_QA]
for q, a in ENV_QA:
    topic1 = (
        q.replace("What is ", "")
        .replace("What are ", "")
        .replace("Why are ", "why ")
        .rstrip("?")
        .lower()
    )
    topic2 = (
        q.replace("What is ", "")
        .replace("What are ", "")
        .replace("What causes ", "")
        .rstrip("?")
        .lower()
    )
    env_base.append({"user": f"Can you explain {topic1}?", "assistant": a})
    env_base.append({"user": f"Tell me about {topic2}.", "assistant": a})
all_examples.extend(diversify_examples(env_base, 400))
print(f"  Environment & Nature: 400 examples")

# --- DATA TRIMMING & SHUFFLING LOGIC ---
random.shuffle(all_examples)
TARGET = 10_000

if len(all_examples) > TARGET:
    identity_subset = [d for d in all_examples if check_identity(d)]
    non_identity_subset = [d for d in all_examples if not check_identity(d)]
    random.shuffle(non_identity_subset)
    all_examples = (
        identity_subset + non_identity_subset[: TARGET - len(identity_subset)]
    )
    random.shuffle(all_examples)

total = len(all_examples)
identity_total = sum(1 for d in all_examples if check_identity(d))

# --- FINAL FILE OUTPUT ---
print(f"\nTotal: {total} examples")
print(f"Identity mentions: {identity_total} ({identity_total / total * 100:.1f}%)")
print(
    f"Non-identity: {total - identity_total} ({(total - identity_total) / total * 100:.1f}%)"
)

with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)

print(f"\nSaved to training_data.json ({total} examples)")
