from crewai import Agent, Task, Tool
from langchain_maistrai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()
class TripAgents:
    def __init__(self):
        self.llm = ChatMistralAI(api_key=os.getenv("MISTRAI_API_KEY"))
    
    def citySelectorAgent(self):
        agent = Agent(
            role = "City Selection Expert",
            goal = "Identify best cities to visit based on user preferences",
            backstory = "An expert travel guide who has extensive knowledge about world cities and their attractions, culture, history, heritage, food, festivals.",
            llm = self.llm,
            verbose = True
        )
        return agent
    
        def localExpertAgent(self):
            agent = Agent(
                role = "Local Destination Expert",
                goal = "Provide information about local attractions, culture, history, heritage, food, festivals.",
                backstory = "A local guide who has first-hand knowledge about the city and its attractions, culture, history, heritage, food, festivals." ",
                llm = self.llm,
                verbose = True
            )

        def budgetAgent(self):
            agent = Agent(
                role = "Budget Planner",
                goal = "Design a trip that stays within a given budget, maximizing its enjoyment and ensuring the best possible experience."
                backstory = "A financial expert who has experience in budgeting and planning trips, ensuring that the user's budget is respected while still providing a great experience."
                llm = self.llm,
                verbose = True
            )
            return agent
        
        def tripPlannerAgent(self):
            agent = Agent(
                role="Trip Planner",
                goal="Design personalized travel itineraries based on user preferences, ensuring a seamless and enjoyable experience.",
                backstory="A seasoned travel expert with a deep understanding of diverse destinations, activities, and cultures. Equipped with the skills to craft itineraries that align with user interests, travel goals, and logistical constraints.",
                llm=self.llm,
                verbose=True
            )
            return agent



class TripTasks:
    def __init__(self):
        pass

    def citySelectionTask(self, agent, inputs):
        task = Agent(
            role = "City Selection",
            goal = f"Select the most suitable city based on user preferences and available {inputs}.",
            description = (
                f"Analyze user preferences and available {inputs} to determine the best city for the trip.\n"
                f"-Travel_type : {inputs['travel_type']}\n"
                f"-Interests : {inputs['interests']}\n"
                f"-Season: {inputs['season']}\n "
                "Output: Suggest three city options that align with the user's preferences and provided {inputs}, along with a concise rationale for each choice."
            ),
            agent = agent,
            expected_output = "Bullet-pointed list of three city options with rationale for each choice."
            
        )
        return task
    
    def cityResearchTask(self, city, agent):
    task = Agent(
        role="City Researcher",
        goal=f"Research and provide detailed information about {city} based on user preferences.",
        description=(
            f"Conduct research on {city} to gather information that matches the user's preferences.\n"
            f"- Key attractions and activities in {city}\n"
            f"- Best travel season for {city}\n"
            f"- Cultural, historical, or culinary highlights in {city}\n"
            f"- Must-try cuisines and local delicacies in {city}\n"
            f"Output: A detailed summary of why {city} is a suitable destination, tailored to the user's interests and must-try cuisines."
        ),
        agent=agent,
        expected_output="A structured report summarizing key attractions, the best travel season, highlights of the city, hotel recommendations, and must-try cuisines."
    )
    return task

class TripCrew:
    pass