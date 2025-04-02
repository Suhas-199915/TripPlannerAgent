from phi.agentimport Agent, Task, Tool  # Importing PHI instead of CrewAI
from langchain_maistrai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()

class TripAgents:
    def __init__(self):
        self.llm = ChatMistralAI(api_key=os.getenv("MISTRAI_API_KEY"))
    
    def citySelectorAgent(self):
        agent = Agent(
            description="You are an expert travel guide specializing in city selection for travelers.",
            instructions=[
                "Analyze user preferences and recommend the most suitable cities.",
                "Consider factors like attractions, culture, food, and festivals."
            ],
            markdown=True,
            debug_mode=True,
            llm=self.llm
        )
        return agent
    
    def localExpertAgent(self):
        agent = Agent(
            description="You are a local destination expert providing valuable insights about cities.",
            instructions=[
                "Share in-depth knowledge about local attractions, history, culture, food, and festivals.",
                "Provide tips for exploring the city like a local."
            ],
            markdown=True,
            debug_mode=True,
            llm=self.llm
        )
        return agent

    def budgetAgent(self):
        agent = Agent(
            description="You are a financial expert who plans trips within a given budget.",
            instructions=[
                "Optimize travel experiences while staying within the user's budget.",
                "Consider expenses for accommodation, food, transport, and activities."
            ],
            markdown=True,
            debug_mode=True,
            llm=self.llm
        )
        return agent
    
    def tripPlannerAgent(self):
        agent = Agent(
            description="You are a seasoned travel planner designing tailored itineraries.",
            instructions=[
                "Create personalized itineraries based on user preferences.",
                "Ensure seamless travel experiences while covering key attractions, meals, and transportation."
            ],
            markdown=True,
            debug_mode=True,
            llm=self.llm
        )
        return agent


class TripTasks:
    def __init__(self):
        pass

    def citySelectionTask(self, agent, inputs):
        task = Task(
            role="City Selection",
            goal=f"Select the most suitable city based on user preferences and available {inputs}.",
            description=(
                f"Analyze user preferences and available {inputs} to determine the best city for the trip.\n"
                f"- Travel Type: {inputs['travel_type']}\n"
                f"- Interests: {inputs['interests']}\n"
                f"- Season: {inputs['season']}\n"
                "Output: Suggest three city options that align with the user's preferences and provided {inputs}, along with a concise rationale for each choice."
            ),
            agent=agent,
            expected_output="A bullet-pointed list of three city options with rationales for each choice."
        )
        return task
    
    def cityResearchTask(self, city, agent):
        task = Task(
            role="City Researcher",
            goal=f"Research and provide detailed information about {city} based on user preferences.",
            description=(
                f"Conduct research on {city} to gather information that matches user preferences.\n"
                f"- Key attractions and activities in {city}\n"
                f"- Best travel season for {city}\n"
                f"- Cultural, historical, or culinary highlights in {city}\n"
                f"- Must-try cuisines and local delicacies in {city}\n"
                "Output: A detailed summary of why {city} is a suitable destination, tailored to the user's interests and must-try cuisines."
            ),
            agent=agent,
            expected_output="A structured report summarizing key attractions, the best travel season, highlights, and must-try cuisines for {city}."
        )
        return task

    def itineraryCreationTask(self, city, agent, inputs):
        task = Task(
            role="Itinerary Creator",
            goal=f"Design a comprehensive travel itinerary for {city} based on user preferences and constraints.",
            description=(
                f"Develop a well-structured itinerary for {city}, ensuring it aligns with user preferences, including:\n"
                f"- Key attractions and must-visit locations in {city}\n"
                f"- Suggested activities for each day, considering time constraints and the provided duration ({inputs['duration']} days)\n"
                f"- Recommendations for meals and must-try cuisines\n"
                f"- Accommodation details, including budget and location suitability\n"
                f"- Transportation tips within {city}\n"
                f"- Special experiences unique to {city}\n"
                "Output: A day-by-day itinerary for {inputs['duration']} days that provides a seamless travel experience, tailored to the user's interests and requirements."
            ),
            agent=agent,
            expected_output=f"A detailed {inputs['duration']}-day itinerary with activities, dining options, and logistical recommendations for {city}."
        )
        return task
    
    def budgetPlanningTask(self, city, agent, inputs, itinerary):
        task = Task(
            role="Budget Planner",
            goal=f"Optimize the budget for a trip to {city}, ensuring the best experience while staying within the given financial constraints.",
            description=(
                f"Plan the budget for a trip to {city}, taking into account:\n"
                f"- Total budget: {inputs['total_budget']}\n"
                f"- Duration: {inputs['duration']} days\n"
                f"- Accommodation preferences: {inputs['accommodation_type']}\n"
                f"- Transport preferences: {inputs['transport_type']}\n"
                f"- Expected expenses for food, activities, and miscellaneous costs\n"
                "Output: A detailed budget breakdown that allocates expenses for accommodation, transport, food, activities, and extras, ensuring optimal use of the available budget."
            ),
            agent=agent,
            context=[itinerary],
            expected_output="A structured budget plan with detailed expense allocations for accommodation, transport, food, activities, and miscellaneous costs."
        )
        return task