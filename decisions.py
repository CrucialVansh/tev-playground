"""CLINC150 decisions shared by every model.

Each request is routed in two steps that every model takes the same way:
first one of ten topics or out of scope, then one of that topic's fifteen
intents or none of them. No question has more than sixteen options, which
keeps Tev (24 at most) and the Laya Core ML export (32 at most) on the same
task as Jev and GPT-4o. The topic grouping is CLINC's own domains.json.
"""

from dataclasses import dataclass
import json
import re
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Label:
    key: str
    name: str
    description: str


OUT_OF_SCOPE = "oos"

TOPICS: tuple[Label, ...] = (
    Label("banking", "Banking", "Bank accounts: balances, transfers, bills, transactions, fraud, checks, and routing numbers."),
    Label("credit_cards", "Credit cards", "Credit cards and credit: limits, rewards, APR, card replacement, applications, and credit scores."),
    Label("kitchen_and_dining", "Kitchen and dining", "Cooking, recipes, nutrition, and restaurants, including reservations and reviews."),
    Label("home", "Home", "Personal organization and the home: music, reminders, calendars, lists, online orders, and smart devices."),
    Label("auto_and_commute", "Auto and commute", "Cars and getting around: maintenance, fuel, tires, traffic, directions, and rides."),
    Label("travel", "Travel", "Trips abroad and bookings: flights, hotels, visas, luggage, currencies, and local customs."),
    Label("utility", "Utility", "Quick everyday tools: time, date, weather, alarms, timers, calls, texts, math, and definitions."),
    Label("work", "Work", "Employment: paid time off, pay, taxes, benefits, meetings, and retirement accounts."),
    Label("small_talk", "Small talk", "Chat with the assistant about itself, greetings, thanks, jokes, and casual conversation."),
    Label("meta", "Assistant settings", "Controlling the assistant itself: its voice, language, name, volume, and short replies like yes, no, or cancel."),
    Label(OUT_OF_SCOPE, "Out of scope", "None of these topics. The request is something this assistant does not handle and should go to a person."),
)

INTENTS: Dict[str, tuple[Label, ...]] = {
    "banking": (
        Label("freeze_account", "Freeze account", "Freeze or lock a bank account."),
        Label("routing", "Routing number", "Get the bank's routing number."),
        Label("pin_change", "Change PIN", "Change a bank card PIN."),
        Label("bill_due", "Bill due date", "When a bill is due."),
        Label("pay_bill", "Pay bill", "Pay a bill."),
        Label("account_blocked", "Account blocked", "A bank account is blocked or locked and the user asks why or how to fix it."),
        Label("interest_rate", "Interest rate", "The interest rate on a bank account."),
        Label("min_payment", "Minimum payment", "The minimum payment due on a bill or card."),
        Label("bill_balance", "Bill balance", "How much is owed on a bill."),
        Label("transfer", "Transfer money", "Move money between accounts or to another person."),
        Label("order_checks", "Order checks", "Order new paper checks."),
        Label("balance", "Account balance", "How much money is in an account."),
        Label("spending_history", "Spending history", "How much was spent on something over a period."),
        Label("transactions", "Recent transactions", "List recent account transactions."),
        Label("report_fraud", "Report fraud", "Report fraudulent or unrecognized activity on an account."),
    ),
    "credit_cards": (
        Label("replacement_card_duration", "Replacement card timing", "How long a replacement card takes to arrive."),
        Label("expiration_date", "Card expiration date", "When a card expires."),
        Label("damaged_card", "Damaged card", "A card is damaged or broken."),
        Label("improve_credit_score", "Improve credit score", "How to raise a credit score."),
        Label("report_lost_card", "Report lost card", "Report a lost or stolen card."),
        Label("card_declined", "Card declined", "Why a card was declined."),
        Label("credit_limit_change", "Change credit limit", "Raise or change a credit limit."),
        Label("apr", "APR", "The APR on a card."),
        Label("redeem_rewards", "Redeem rewards", "How to redeem or use card rewards points."),
        Label("credit_limit", "Credit limit", "What a card's credit limit is."),
        Label("rewards_balance", "Rewards balance", "How many rewards points are available."),
        Label("application_status", "Application status", "Status of a credit card application."),
        Label("credit_score", "Credit score", "What the user's credit score is."),
        Label("new_card", "New card", "Apply for or get a new credit card."),
        Label("international_fees", "International fees", "Fees for using a card abroad or in a foreign currency."),
    ),
    "kitchen_and_dining": (
        Label("food_last", "Food shelf life", "How long a food keeps before it goes bad."),
        Label("confirm_reservation", "Confirm reservation", "Confirm an existing restaurant reservation."),
        Label("how_busy", "How busy", "How busy or crowded a restaurant is, or the wait time."),
        Label("ingredients_list", "Ingredients list", "The ingredients needed for a dish."),
        Label("calories", "Calories", "How many calories a food has."),
        Label("nutrition_info", "Nutrition info", "Nutritional content of a food other than calories."),
        Label("recipe", "Recipe", "How to make a dish."),
        Label("restaurant_reviews", "Restaurant reviews", "Reviews or ratings of a restaurant."),
        Label("restaurant_reservation", "Make reservation", "Book a table at a restaurant."),
        Label("meal_suggestion", "Meal suggestion", "Suggest something to cook or eat."),
        Label("restaurant_suggestion", "Restaurant suggestion", "Recommend a restaurant."),
        Label("cancel_reservation", "Cancel reservation", "Cancel a restaurant reservation."),
        Label("ingredient_substitution", "Ingredient substitute", "What to use instead of an ingredient."),
        Label("cook_time", "Cooking time", "How long to cook something."),
        Label("accept_reservations", "Takes reservations", "Whether a restaurant accepts reservations."),
    ),
    "home": (
        Label("what_song", "What song", "Identify the song that is playing."),
        Label("play_music", "Play music", "Play a song, artist, or genre."),
        Label("todo_list_update", "Update to-do list", "Add to or remove from the to-do list."),
        Label("reminder", "Check reminders", "List or read existing reminders."),
        Label("reminder_update", "Set reminder", "Create or change a reminder."),
        Label("calendar_update", "Update calendar", "Add, move, or remove a calendar event."),
        Label("order_status", "Order status", "Where an online order or delivery is."),
        Label("update_playlist", "Update playlist", "Add songs to or change a playlist."),
        Label("shopping_list", "Check shopping list", "Read what is on the shopping list."),
        Label("calendar", "Check calendar", "What is on the calendar."),
        Label("next_song", "Next song", "Skip to the next song."),
        Label("order", "Place order", "Order or buy something online."),
        Label("todo_list", "Check to-do list", "Read what is on the to-do list."),
        Label("shopping_list_update", "Update shopping list", "Add to or remove from the shopping list."),
        Label("smart_home", "Smart home", "Control a smart home device such as lights, thermostat, or appliances."),
    ),
    "auto_and_commute": (
        Label("current_location", "Current location", "Where the user is right now."),
        Label("oil_change_when", "When to change oil", "When the car's next oil change is due."),
        Label("oil_change_how", "How to change oil", "How to change the car's oil."),
        Label("uber", "Book a ride", "Book an Uber or other ride."),
        Label("traffic", "Traffic", "Traffic conditions on a route."),
        Label("tire_pressure", "Tire pressure", "The car's tire pressure."),
        Label("schedule_maintenance", "Schedule maintenance", "Book car maintenance or service."),
        Label("gas", "Fuel level", "How much fuel is in the car."),
        Label("mpg", "Fuel economy", "The car's miles per gallon."),
        Label("distance", "Distance", "How far away a place is or how long it takes to get there."),
        Label("directions", "Directions", "Directions to a place."),
        Label("last_maintenance", "Last maintenance", "When the car was last serviced."),
        Label("gas_type", "Fuel type", "What kind of fuel the car takes."),
        Label("tire_change", "Tire change", "When or how to change a tire."),
        Label("jump_start", "Jump start", "How to jump start a car."),
    ),
    "travel": (
        Label("plug_type", "Plug type", "What electrical plug or adapter is used in a country."),
        Label("travel_notification", "Travel notice", "Tell the bank or card issuer about upcoming travel."),
        Label("translate", "Translate", "Translate a phrase into another language."),
        Label("flight_status", "Flight status", "Status or time of a flight."),
        Label("international_visa", "Visa", "Whether a visa is needed to visit a country."),
        Label("timezone", "Time zone", "The time zone of a place."),
        Label("exchange_rate", "Exchange rate", "The exchange rate between currencies."),
        Label("travel_suggestion", "Travel suggestion", "Recommend places to visit or things to do."),
        Label("travel_alert", "Travel alert", "Travel safety warnings or advisories for a place."),
        Label("vaccines", "Vaccines", "Vaccinations needed for a trip."),
        Label("lost_luggage", "Lost luggage", "Luggage is lost or missing."),
        Label("book_flight", "Book flight", "Book a flight."),
        Label("book_hotel", "Book hotel", "Book a hotel."),
        Label("carry_on", "Carry-on rules", "Carry-on baggage rules or limits."),
        Label("car_rental", "Car rental", "Rent a car."),
    ),
    "utility": (
        Label("weather", "Weather", "The weather or forecast."),
        Label("alarm", "Alarm", "Set or check an alarm."),
        Label("date", "Date", "Today's date or the date of a day."),
        Label("find_phone", "Find phone", "Locate the user's phone."),
        Label("share_location", "Share location", "Share the user's location with someone."),
        Label("timer", "Timer", "Set a timer."),
        Label("make_call", "Make call", "Call someone."),
        Label("calculator", "Calculator", "Do arithmetic."),
        Label("definition", "Definition", "The meaning of a word."),
        Label("measurement_conversion", "Unit conversion", "Convert between units of measurement."),
        Label("flip_coin", "Flip coin", "Flip a coin."),
        Label("spelling", "Spelling", "How to spell a word."),
        Label("time", "Time", "The current time."),
        Label("roll_dice", "Roll dice", "Roll dice."),
        Label("text", "Send text", "Send a text message."),
    ),
    "work": (
        Label("pto_request_status", "Time-off request status", "Whether a time-off request was approved."),
        Label("next_holiday", "Next holiday", "When the next holiday is."),
        Label("insurance_change", "Change insurance", "Change health or other insurance coverage."),
        Label("insurance", "Insurance", "What the user's insurance plan or benefits are."),
        Label("meeting_schedule", "Check meetings", "What meetings are scheduled."),
        Label("payday", "Payday", "When the next paycheck arrives."),
        Label("taxes", "Taxes", "Taxes owed, filing, or refunds."),
        Label("income", "Income", "How much the user earns."),
        Label("rollover_401k", "401k rollover", "Roll over a 401k retirement account."),
        Label("pto_balance", "Time-off balance", "How many vacation or time-off days are left."),
        Label("pto_request", "Request time off", "Ask for time off or vacation days."),
        Label("w2", "W-2 form", "Get a W-2 tax form."),
        Label("schedule_meeting", "Schedule meeting", "Book a meeting."),
        Label("direct_deposit", "Direct deposit", "Set up or change direct deposit of pay."),
        Label("pto_used", "Time off used", "How many time-off days have been used."),
    ),
    "small_talk": (
        Label("who_made_you", "Who made you", "Who built or created the assistant."),
        Label("meaning_of_life", "Meaning of life", "The meaning of life."),
        Label("who_do_you_work_for", "Who do you work for", "Who the assistant works for or serves."),
        Label("do_you_have_pets", "Pets", "Whether the assistant has pets."),
        Label("what_are_your_hobbies", "Hobbies", "What the assistant does for fun."),
        Label("fun_fact", "Fun fact", "Tell a fun fact."),
        Label("what_is_your_name", "Assistant's name", "What the assistant is called."),
        Label("where_are_you_from", "Where from", "Where the assistant is from."),
        Label("goodbye", "Goodbye", "Say goodbye."),
        Label("thank_you", "Thank you", "Thank the assistant."),
        Label("greeting", "Greeting", "Say hello or ask how the assistant is."),
        Label("tell_joke", "Joke", "Tell a joke."),
        Label("are_you_a_bot", "Are you a bot", "Whether the assistant is a bot or a person."),
        Label("how_old_are_you", "Age", "How old the assistant is."),
        Label("what_can_i_ask_you", "Capabilities", "What the assistant can help with."),
    ),
    "meta": (
        Label("change_speed", "Change speed", "Make the assistant talk faster or slower."),
        Label("user_name", "User's name", "What name the assistant uses for the user."),
        Label("whisper_mode", "Whisper mode", "Make the assistant whisper or speak quietly."),
        Label("yes", "Yes", "The user says yes or agrees."),
        Label("change_volume", "Change volume", "Make the assistant louder or quieter."),
        Label("no", "No", "The user says no or disagrees."),
        Label("change_language", "Change language", "Make the assistant speak another language."),
        Label("repeat", "Repeat", "Ask the assistant to repeat what it said."),
        Label("change_accent", "Change accent", "Change the assistant's accent."),
        Label("cancel", "Cancel", "Cancel or stop the current request."),
        Label("sync_device", "Sync device", "Sync or pair the assistant with a device."),
        Label("change_user_name", "Change user's name", "Change what the assistant calls the user."),
        Label("change_ai_name", "Rename assistant", "Give the assistant a new name."),
        Label("reset_settings", "Reset settings", "Reset the assistant to factory settings."),
        Label("maybe", "Maybe", "The user is unsure or says maybe."),
    ),
}

NONE_OF_THESE = Label(OUT_OF_SCOPE, "None of these", "None of these fit. The request is something else and should go to a person.")

TOPIC_QUESTION = "Which topic does this request to a virtual assistant belong to?"
INTENT_QUESTION = "The request is about {topic}. What does the user want?"

TOPIC_BY_INTENT: Dict[str, str] = {intent.key: topic for topic, intents in INTENTS.items() for intent in intents}
TOPIC_BY_KEY: Dict[str, Label] = {label.key: label for label in TOPICS}

MAX_TEXT_CHARS = 2000
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def intent_options(topic: str) -> tuple[Label, ...]:
    return INTENTS[topic] + (NONE_OF_THESE,)


def intent_question(topic: str) -> str:
    return INTENT_QUESTION.format(topic=TOPIC_BY_KEY[topic].name.lower())


def _normalize_token(value: str) -> str:
    token = value.strip().lower().replace("&", " and ")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def aliases(labels: Sequence[Label]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for label in labels:
        for raw in (label.key, label.name):
            mapping[_normalize_token(raw)] = label.key
    return mapping


def resolve_label(value: Optional[str], mapping: Mapping[str, str]) -> Optional[str]:
    if value is None:
        return None
    token = _normalize_token(str(value))
    if not token:
        return None
    return mapping.get(token)


def query_text(text: object) -> Optional[str]:
    value = "" if text is None else str(text).strip()
    if not value:
        return None
    return value[:MAX_TEXT_CHARS]


def choice_question(question: str, labels: Sequence[Label]) -> dict:
    return {
        "type": "choice",
        "instructions": question,
        "criteria": {label.key: label.description for label in labels},
    }


def tev_options(labels: Sequence[Label]) -> List[dict]:
    if not 2 <= len(labels) <= 24:
        raise ValueError(f"Tev accepts 2 to 24 options, got {len(labels)}.")
    return [
        {"label": OPTION_LETTERS[index], "key": label.key, "description": label.description}
        for index, label in enumerate(labels)
    ]


def tev_task(state: str, question: str, labels: Sequence[Label]) -> str:
    payload = {"state": state, "question": question, "options": tev_options(labels)}
    return json.dumps(payload, ensure_ascii=False)


def parse_option_letter(raw: str, labels: Sequence[Label]) -> Optional[str]:
    if not raw:
        return None
    allowed = {OPTION_LETTERS[index]: label.key for index, label in enumerate(labels)}
    match = re.search(r"\b([A-Z])\b", raw.upper())
    if match and match.group(1) in allowed:
        return allowed[match.group(1)]
    return resolve_label(raw, aliases(labels))
