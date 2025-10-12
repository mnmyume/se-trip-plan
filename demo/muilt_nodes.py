import dspy
from agents.server_llm import serverLLM


lm = serverLLM(
    model_name="gemma-3-27b-it",
    api_url="https://django.cair.mun.ca/llm/v1/chat/completions",
    api_key="ADAjs78ehDSS87hs3edcf4edr5"
)
dspy.configure(lm=lm)


class TransportationPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.OutputField()

class AttractionPlan(dspy.Signature):
    user_request = dspy.InputField()
    attraction = dspy.OutputField()

class AccommodationPlan(dspy.Signature):
    user_request = dspy.InputField()
    attraction = dspy.InputField()
    accommodation = dspy.OutputField()

class RestaurantPlan(dspy.Signature):
    user_request = dspy.InputField()
    attraction = dspy.InputField()
    restaurant = dspy.OutputField()

class CombinedPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.InputField()
    attraction = dspy.InputField()
    accommodation = dspy.InputField()
    restaurant = dspy.InputField()
    final_plan = dspy.OutputField()


class TripPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm = lm  # 用传入的 LLM

    def forward(self, **kwargs):
        user_req = kwargs["user_request"]

        # 1️⃣ Attraction
        attraction_msg = [
            {"role": "system", "content": "You are a helpful travel planner."},
            {"role": "user", "content": f"List must-see attractions for: {user_req}"}
        ]
        attraction_resp = self.lm(messages=attraction_msg)
        attraction = getattr(attraction_resp, "content", str(attraction_resp))

        # 2️⃣ Transportation
        transport_msg = [
            {"role": "system", "content": "You are a transportation expert."},
            {"role": "user", "content": f"Suggest best transportation for: {user_req}"}
        ]
        transport_resp = self.lm(messages=transport_msg)
        transportation = getattr(transport_resp, "content", str(transport_resp))

        # 3️⃣ Accommodation
        accommodation_msg = [
            {"role": "system", "content": "You are a hotel recommendation expert."},
            {"role": "user", "content": f"For {user_req}, near {attraction}, recommend suitable accommodations."}
        ]
        accommodation_resp = self.lm(messages=accommodation_msg)
        accommodation = getattr(accommodation_resp, "content", str(accommodation_resp))

        # 4️⃣ Restaurant
        restaurant_msg = [
            {"role": "system", "content": "You are a local food recommendation expert."},
            {"role": "user", "content": f"For {user_req}, near {attraction}, recommend local restaurants."}
        ]
        restaurant_resp = self.lm(messages=restaurant_msg)
        restaurant = getattr(restaurant_resp, "content", str(restaurant_resp))

        # 5️⃣ Combine
        combine_msg = [
            {"role": "system", "content": "You are an expert trip organizer."},
            {"role": "user", "content": (
                f"Combine all into a coherent 7-day plan:\n"
                f"Transportation: {transportation}\n"
                f"Attractions: {attraction}\n"
                f"Accommodation: {accommodation}\n"
                f"Restaurants: {restaurant}\n"
                f"Create a clear, day-by-day travel plan."
            )}
        ]
        final_resp = self.lm(messages=combine_msg)
        final_plan = getattr(final_resp, "content", str(final_resp))

        return {
            "final_plan": final_plan,
            "transportation": transportation,
            "attraction": attraction,
            "accommodation": accommodation,
            "restaurant": restaurant
        }


planner = TripPlanner()
resp = planner(user_request="Plan a 7-day trip from London to Tokyo with a moderate budget, prefer local food and must-see spots")

print("\n=== OUTPUT ===")
print("Transportation:", resp["transportation"])
print("Attraction:", resp["attraction"])
print("Accommodation:", resp["accommodation"])
print("Restaurant:", resp["restaurant"])
print("Final Plan:", resp["final_plan"])
