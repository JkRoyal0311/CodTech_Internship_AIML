from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization")

# Your input text
text = """
Sure! Here's **600 words of random, meaningful text** you can use for testing your summarization script. It blends multiple topics (technology, education, AI, and general knowledge) to simulate a real-world input:

---

### 📄 **Random 600-Word Text**

Artificial Intelligence (AI) has rapidly evolved over the past few decades and now influences nearly every aspect of modern life. From virtual assistants like Siri and Alexa to self-driving cars and smart home devices, AI is shaping how we interact with technology. At the core of AI are algorithms and machine learning models that allow systems to learn from data, adapt to new inputs, and perform tasks previously done by humans. This advancement has created both excitement and concern, as debates continue about job displacement, ethical considerations, and the long-term impacts of intelligent machines on society.

Education, too, has seen a technological revolution. Online learning platforms such as Coursera, Udemy, and Khan Academy allow students to learn from anywhere in the world. Especially during the COVID-19 pandemic, e-learning became a lifeline for millions of students. Technologies such as virtual reality (VR) and augmented reality (AR) are being integrated into classrooms to create immersive learning experiences. These tools help in teaching complex subjects like anatomy, astronomy, and physics by visualizing concepts in 3D.

Climate change remains a pressing global issue, and technology is playing a vital role in combating it. Innovations in renewable energy, such as solar panels, wind turbines, and battery storage, are helping reduce our dependence on fossil fuels. Governments and private sectors are investing in smart grid systems and electric vehicles to lower carbon emissions. AI is also being used to monitor deforestation, track animal migration, and model climate predictions more accurately than ever before.

Healthcare is another sector undergoing rapid transformation. Telemedicine allows patients to consult doctors remotely, improving access in rural or underserved areas. Wearable devices like smartwatches monitor heart rate, blood oxygen, and sleep patterns, giving users greater control over their health. AI-powered diagnostics can now analyze medical images like X-rays and MRIs to detect conditions such as cancer earlier and with more accuracy.

The financial world is also being reshaped by fintech innovations. Mobile banking, cryptocurrency, and robo-advisors are making financial services more accessible to people around the world. Blockchain technology, initially known for supporting Bitcoin, is now being explored for secure and transparent applications in healthcare, supply chains, and voting systems.

Space exploration continues to inspire awe and innovation. Companies like SpaceX, Blue Origin, and NASA are pushing the boundaries of what is possible beyond Earth. Reusable rockets, lunar missions, and even plans for Mars colonization have moved from science fiction into plausible future endeavors. These missions are not just about reaching other planets but also about developing technologies that benefit life on Earth, such as satellite communication, GPS, and weather forecasting.

Cultural and language preservation is another area seeing growth through technology. Machine translation tools like Google Translate and DeepL allow people from different linguistic backgrounds to communicate. Digital archiving helps preserve endangered languages, traditional songs, and ancient manuscripts, making them accessible to future generations.

Mental health awareness has increased, with mobile apps like Headspace and Calm promoting mindfulness and stress reduction. Chatbots are being trained to provide initial mental health support, making services more available and less stigmatized. However, there is ongoing debate about the limitations of using AI in emotionally sensitive situations.

As we continue into the 21st century, the line between the digital and physical world becomes increasingly blurred. Concepts like the metaverse propose virtual spaces where people can interact, work, and play using avatars. While exciting, this also raises questions about identity, privacy, and digital addiction.

In conclusion, technology offers enormous potential to solve real-world problems, improve lives, and expand human understanding. But it also requires thoughtful consideration, ethical design, and inclusive development to ensure that its benefits are shared by all.

---

Let me know if you'd like:

* Longer data (e.g., 1000+ words)
* Specific topic (e.g., history, sports, science fiction)
* Input from a `.txt` file instead of raw text

"""

# Generate summary
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)

# Print result
print("Summary:\n", summary[0]['summary_text'])
