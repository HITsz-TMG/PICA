# enumerate instruction for think


ICL_CONFIG = {
    "format_dict": {
        "system":
"""# Instruction

Below is a list of conversations between a human and an AI assistant (you). 
As an AI assistant, you will engage in conversations with users, responding to their queries which are presented under the heading "# Query:". 
Your responses should be entered under the heading "# Answer:". 
You excel in a wide range of tasks including, but not limited to, providing general information, conducting reasoning, engaging in role-play, creative writing, planning, and solving mathematical and coding problems. 
Your responses should be well-structured, comprehensive, and aim to thoroughly address the user's query or problem at hand.
When encountering "<|Thinking|>", pause to carefully consider the content before proceeding to ensure clarity and accuracy in your responses.

""",
        "eos": "\n<|Thinking|>\n```\n\n",
        "proj_tokens": "\n```\n\n# Answer:\n```\n<|Thinking|>",
        "query_format": "# Query:\n```\n{}",
    },
    "demons": [
            (
"""What are some famous world music artists?""",
"""World music encompasses a vast range of styles from different cultures around the globe. Here are some famous artists known for their contributions to world music:

1. **Youssou N'Dour** - Senegal
2. **Ali Farka Touré** - Mali
3. **Cesária Évora** - Cape Verde
4. **Ravi Shankar** - India
5. **Ladysmith Black Mambazo** - South Africa
6. **Shakira** - Colombia
7. **Angelique Kidjo** - Benin
8. **Fela Kuti** - Nigeria
9. **Gipsy Kings** - France (of Spanish descent)
10. **Tinariwen** - Mali

These artists have significantly influenced world music with their unique styles and cultural backgrounds.""",
            ),
            (
"""You will act as a digital marketing specialist. What keywords should a novel writing software target?""",
"""To efficiently target keywords for a novel writing software, one must consider the target audience's search habits and the various functionalities or attributes of the software that might attract prospective users. Here's a breakdown of keyword categories and examples that a novel writing software could aim for:

1. **Primary Keywords** - Directly connected to the main product:
   - Novel writing software
   - Book writing software
   - Novel authoring application
   - Book creation app
2. **Feature-Specific Keywords** - Emphasizing special or crucial features of the software:
   - Novel plotting software
   - Collaborative writing tool for authors
   - Real-time editing software for writers
   - Character building tool for novelists
   - Setting and scene development software
3. **User Intent Keywords** - Based on the goals of potential users:
   - Best software for writing a novel
   - How to write a book
   - Novel writing software for amateurs
   - Advanced book writing software
   - Free novel writing app
4. **Platform-Specific Keywords** - If the software operates on certain platforms or devices:
   - Novel writing software for PC
   - Book writing app for macOS
   - Writing software for Android tablets
   - Mobile app for book writers
5. **Competitor Keywords** - Names of or comparisons with competing novel writing software:
   - Scrivener alternatives
   - Ulysses vs. [Your Software Name]
   - Novel writing software similar to yWriter
   - Features comparison with Novel Factory
6. **Industry-Specific Keywords** - Pertaining to the book writing and publishing industry:
   - Publishing industry writing tools
   - Fiction writing software
   - Standard book writing software
   - Software for novelists and fiction writers
7. **Educational and Supportive Content Keywords** - For articles, guides, and support content:
   - How to plot a novel
   - Tips for first-time novelists
   - Best novel writing practices
   - Novel writing software guides
   - Book writing online courses
8. **Long-Tail Keywords** - More specific phrases that often face less competition:
   - Affordable writing software for independent authors
   - Best character development tool for novelists
   - Writing software with timeline features for plot management
9. **Local/Regional Keywords** - Targeting users in specific locations:
   - Novel writing software in New York
   - Book writing apps in the UK
10. **Event or Trend-Related Keywords** - Relevant to current industry events or trends:
    - Award-winning authors' software choices
    - Novel writing software for National Novel Writing Month participants
When choosing keywords, it's essential to perform detailed keyword research using tools such as Google Keyword Planner, SEMrush, Ahrefs, or Moz to gauge search volume, competition intensity, and relevancy. It's also important to understand how the intended audience might describe their needs or search for solutions, which will help shape an effective keyword strategy. Monitoring keyword trends and adapting strategies based on market and search behavior changes are crucial for maintaining relevance and effectiveness."""
            ),
            (
"""Is the US border open to Mexico?""",
"""As of my last update in March 2023, the United States border with Mexico is open, but entry requirements are in place due to ongoing COVID-19 management efforts. These entry requirements are subject to change in response to the pandemic's dynamics, so it's crucial to check the most recent information before making travel arrangements.
Typically, travelers might be required to show proof of vaccination, present a negative COVID-19 test result, or adhere to other conditions established by the U.S. government. Since these restrictions can be updated frequently and sometimes abruptly, it is recommended to refer to authoritative sources such as the U.S. Department of State, U.S. Customs and Border Protection (CBP), or the Centers for Disease Control and Prevention (CDC) for the latest travel guidelines.
To stay informed about the most current rules, you should:

1. Visit the official websites of the U.S. Department of Homeland Security or U.S. Customs and Border Protection.
2. Consult the latest guidelines issued by the Centers for Disease Control and Prevention (CDC).
3. Check for any advisories or updates from the U.S. Department of State.
4. If you are flying, also verify any additional protocols with your airline.

Additionally, remember to check the entry requirements for returning to Mexico from the United States, as Mexican authorities might have distinct regulations for entry or re-entry."""
            ),
            (
"""Look for poems that mention the given object or character. The names of the poems and their poets should be written.\n\nthe moon""",
"""1. "The Moon and the Yew Tree" by Sylvia Plath
2. "The Moon" by Robert Louis Stevenson
3. "Above the Dock" by T. E. Hulme
4. "Moonrise" by D. H. Lawrence
5. "The Freedom of the Moon" by Robert Frost
6. "To the Moon" by Percy Bysshe Shelley
7. "The Cat and the Moon" by W. B. Yeats

These are just a few examples of poems that mention the moon. This celestial body is a frequent motif in poetry, symbolizing various themes such as beauty, mystery, and the passage of time, making it a beloved subject for many poets."""
            ),
        ]
}
