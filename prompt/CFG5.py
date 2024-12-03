ICL_CONFIG = {
    "format_dict": {
        "system":
"""# Instruction

Below is a list of conversations between a human and an AI assistant (you). 
As an AI assistant, you will engage in conversations with users, responding to their queries which are presented under the heading "# Query:". 
Your responses should be entered under the heading "# Answer:". 
You excel in a wide range of tasks including, but not limited to, providing general information, conducting reasoning, engaging in role-play, creative writing, planning, and solving mathematical and coding problems. 
Your responses should be well-structured, comprehensive, and aim to thoroughly address the user's query or problem at hand.

""",
        "eos": "\n```\n\n",
        "proj_tokens": "\n```\n\n# Answer:\n```\n",
        "query_format": "# Query:\n```\n{}",
    },
    "demons": [
            (
"""Tom has two sons, Mike and Matt. Matt has one brother and a father. The father has two sons. Matt's brother has a brother and a father. Who is Matt's brother?""",
"""Matt's brother is Mike. Matt has one brother and a father, and since Tom has two sons, Mike and Matt, it is clear that they are each other's only brother. Therefore, when you say "Matt's brother has a brother and a father," you are describing Mike, who indeed has a brother (Matt) and a father (Tom)."""
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
"""68-year-old male with diabetes for about six to seven years (HbA1c consistently between 5.8-6.4, fasting blood glucose around 7.2, other times elevated between 8-9-9.4, no episodes of hypoglycemia). CKD for four years (initially with 24-hour urine protein under 250, GFR around 95, but in the last 1-2 years urine protein between 350-450mg, GFR between 65-75, most recent one being 59). No pain, not obese, blood pressure within normal ranges, but often experiences itchy skin during winter; regularly suffers from constipation throughout the year. <br><br>Current medication: 1. Losartan Tablets 240mg/day 2. Empagliflozin Tablets 10mg/day 3. Metformin HCL 500mg*3/day 4. Digestive enzyme supplements.<br><br>Are there more advanced treatment options and medications available? Can dialysis be postponed or avoided? Are there other tests that need to be performed? What are the recommendations for exercise and diet besides medication? When should the current medications be switched to insulin due to CKD?<br>""",
"""Treatment options should be individualized and overseen by healthcare professionals following a comprehensive assessment. However, here is some general guidance on treatment options and considerations for an individual with diabetes and chronic kidney disease (CKD).

**Medications:**
1. **Losartan** is an angiotensin II receptor antagonist used to control high blood pressure and can aid in protecting renal function in CKD.
2. **Empagliflozin** is a sodium-glucose cotransporter 2 (SGLT2) inhibitor, which helps manage blood sugar levels and also offers cardiovascular and renal benefits.
3. **Metformin** is a primary treatment for type 2 diabetes but should be used cautiously in CKD due to potential lactic acidosis risk, especially as renal function declines.
4. **Digestive enzyme supplements** are typically used to aid digestion but aren't directly related to managing diabetes or CKD.

**Potential Treatment Adjustments:**
- **Regular renal function monitoring** is critical. Metformin may need discontinuation or dosage adjustments if the GFR continues to drop to prevent lactic acidosis.
- **Consideration of newer medications** like GLP-1 receptor agonists (e.g., liraglutide, semaglutide) which are effective in glycemic control and cardiovascular risk reduction and generally safe in CKD.
- **Transitioning to insulin therapy** may become necessary if glucose levels are unmanageable with oral treatments or if renal health declines significantly.

**Additional Exams:**
- **Frequent checks** of renal function (eGFR) and protein in the urine.
- **Examinations for diabetes-related complications** such as eye, nerve, and heart conditions.
- **Blood tests** for electrolytes (including potassium and phosphorus), full blood count (CBC), lipid profile, and parathyroid hormone (PTH) levels.

**Diet and Exercise:**
- **Diet management** is crucial, possibly including a diet low in sodium, phosphorus, and protein to help slow CKD progression.
- **Managing blood sugar** through methods like carbohydrate counting or a consistent carbohydrate intake can be beneficial.
- **Regular exercise** is advised, such as 150 minutes of moderate-intensity activity per week, unless contraindicated.

**Lifestyle Adjustments:**
- To alleviate itching, possibly related to dry skin or elevated phosphorus, use suitable skin moisturizers and monitor phosphorus levels.
- Constipation may be relieved by increasing dietary fiber, hydration, and potentially gentle laxatives, ensuring these are appropriate given the CKD condition.

**Indications for Insulin Therapy:**
- Insulin might be initiated when oral medications can't adequately control blood glucose, during serious illness, or when renal function is severely impaired, making other diabetes medications contraindicated or necessitating very cautious use.

Continuous follow-up with healthcare providers is crucial to tailor treatment plans as needed, considering CKD progression and other personal health factors. Coordination among the primary care physician, endocrinologist, and nephrologist is vital for optimal management of both diabetes and CKD."""
            ),
            (
"""Given a sentence that may be ambiguous, rewrite the sentence using less ambiguous words.\n\nThe chicken is ready to eat.""",
"""The chicken has been cooked and is ready to be eaten."""
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
