import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from openenv.core import Environment, Action, Observation, State

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    is_spam: bool = False
    is_high_priority: bool = False

class TriageAction(Action):
    action_type: str = Field(description="One of: 'read', 'reply', 'escalate', 'mark_spam', 'finish'")
    ticket_id: Optional[str] = Field(None, description="Ticket ID for actions: 'reply', 'escalate', 'mark_spam', 'read'")
    reply_text: Optional[str] = Field(None, description="Text if replying")

class TriageObservation(Observation):
    echoed_message: str = Field(description="Message indicating the result of the action")
    current_ticket_content: Optional[str] = Field(None, description="Content of the read ticket")
    inbox_status: str = Field(description="Summary of inbox status")

class TriageState(State):
    tickets: Dict[str, Ticket] = Field(default_factory=dict)
    processed_tickets: List[str] = Field(default_factory=list)
    escalated_tickets: List[str] = Field(default_factory=list)
    replied_tickets: Dict[str, str] = Field(default_factory=dict)
    spam_tickets: List[str] = Field(default_factory=list)
    difficulty: str = Field("easy")

class SupportTriageEnv(Environment[TriageAction, TriageObservation, TriageState]):
    """
    Real-world environment for Support Ticket Triage.
    The agent acts as a support representative who must read incoming tickets,
    reply to standard ones, escalate high-priority/urgent ones, and mark spam.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = TriageState()
        self._done = False
        self._difficulty = "easy"
        self._reward_scale = 1.0

    @property
    def state(self) -> TriageState:
        return self._state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_difficulty: str = "easy", **kwargs) -> TriageObservation:
        self._difficulty = task_difficulty
        self._state = TriageState(difficulty=task_difficulty)
        self._done = False
        
        # Grading / tasks
        if task_difficulty == "easy":
            # 1 normal, 1 spam (Task 1)
            self._state.tickets = {
                "t1": Ticket(id="t1", subject="Login issue", body="I can't login.", sender="user@example.com"),
                "t2": Ticket(id="t2", subject="YOU WON!!!", body="Click here to claim your millions!", sender="scam@scam.com", is_spam=True)
            }
        elif task_difficulty == "medium":
            # 2 normal, 1 spam, 1 high priority (Task 2)
            self._state.tickets = {
                "t1": Ticket(id="t1", subject="Refund please", body="I want a refund for order #123", sender="bob@test.com"),
                "t2": Ticket(id="t2", subject="Server is down!", body="Production is down! ALL endpoints 500.", sender="urgent@sys.com", is_high_priority=True),
                "t3": Ticket(id="t3", subject="Update billing", body="How do I update card?", sender="alice@test.com"),
                "t4": Ticket(id="t4", subject="Cheap deals", body="Buy now!", sender="spam@spam.com", is_spam=True)
            }
        else: # hard
            # 2 normal, 2 spam, 1 high priority (Task 3)
            self._state.tickets = {
                "t1": Ticket(id="t1", subject="API docs", body="Where are the docs for bulk create?", sender="dev@test.com"),
                "t2": Ticket(id="t2", subject="Security flaw", body="We found a critical vulnerability allowing remote execution.", sender="security@test.com", is_high_priority=True),
                "t3": Ticket(id="t3", subject="Marketing", body="I guarantee 10k leads", sender="marketing@spam.com", is_spam=True),
                "t4": Ticket(id="t4", subject="Forgot user", body="What is my username?", sender="john@test.com"),
                "t5": Ticket(id="t5", subject="Nigerian Prince", body="10 million dollars", sender="prince@spam.com", is_spam=True)
            }

        # To ensure the total reward over the episode sums up perfectly to 0.0 - 1.0 (as required by grader)
        # Each correctly processed ticket grants (1.0 / num_tickets)
        self._reward_scale = 1.0 / len(self._state.tickets)
        
        obs = self._make_observation("Environment reset. Inbox loaded.", 0.0)
        return obs

    def _make_observation(self, msg: str, reward: float, current_ticket: Optional[str] = None) -> TriageObservation:
        inbox_count = len(self._state.tickets)
        processed = len(self._state.processed_tickets)
        status = f"Inbox: {inbox_count - processed} remaining. Processed: {processed}/{inbox_count}."
        # Clamp reward to 0..1 globally is done across the episode, but a single step can be negative
        
        return TriageObservation(
            echoed_message=msg,
            current_ticket_content=current_ticket,
            inbox_status=status,
            done=self._done,
            reward=reward
        )

    def step(self, action: TriageAction, timeout_s: Optional[float] = None, **kwargs) -> TriageObservation:
        if self._done:
            return self._make_observation("Episode is already done.", 0.0)
            
        action_type = action.action_type.lower()
        ticket_id = action.ticket_id
        
        if action_type == "finish":
            self._done = True
            return self._make_observation("Finished triage.", 0.0)
            
        if not ticket_id or ticket_id not in self._state.tickets:
            # Minor penalty for invalid actions
            return self._make_observation(f"Invalid ticket ID: {ticket_id}", -0.05 * self._reward_scale)
            
        if ticket_id in self._state.processed_tickets and action_type != "read":
            return self._make_observation(f"Ticket {ticket_id} is already processed.", -0.05 * self._reward_scale)

        t = self._state.tickets[ticket_id]
        reward_fraction = 0.0
        msg = ""
        current_ticket = None

        if action_type == "read":
            msg = f"Reading ticket {ticket_id}"
            current_ticket = f"From: {t.sender}\nSubject: {t.subject}\nBody: {t.body}"
            # No direct reward for reading, it's just to gather info
                
        elif action_type == "reply":
            if t.is_spam:
                reward_fraction = 0.0  # complete fail for this ticket
                msg = f"Replied to spam ticket {ticket_id}. Incorrect."
            elif t.is_high_priority:
                reward_fraction = 0.2  # partial credit for replying instead of escalating
                msg = f"Replied to high priority ticket {ticket_id} instead of escalating."
            else:
                reward_fraction = 1.0  # Perfect
                msg = f"Properly replied to normal ticket {ticket_id}."

            self._state.replied_tickets[ticket_id] = action.reply_text or ""
            self._state.processed_tickets.append(ticket_id)
                
        elif action_type == "escalate":
            if t.is_spam:
                reward_fraction = 0.0
                msg = f"Escalated spam ticket {ticket_id}. Incorrect."
            elif not t.is_high_priority:
                reward_fraction = 0.0
                msg = f"Escalated normal ticket {ticket_id} unnecessarily."
            else:
                reward_fraction = 1.0  # Perfect
                msg = f"Properly escalated high priority ticket {ticket_id}."
                
            self._state.escalated_tickets.append(ticket_id)
            self._state.processed_tickets.append(ticket_id)
                
        elif action_type == "mark_spam":
            if not t.is_spam:
                reward_fraction = 0.0
                msg = f"Marked legitimate ticket {ticket_id} as spam. Incorrect."
            else:
                reward_fraction = 1.0  # Perfect
                msg = f"Correctly marked {ticket_id} as spam."
                
            self._state.spam_tickets.append(ticket_id)
            self._state.processed_tickets.append(ticket_id)
        else:
            msg = f"Unknown action: {action_type}"

        if len(self._state.processed_tickets) == len(self._state.tickets):
            self._done = True
            msg += " All tickets processed. Episode done."
            
        final_step_reward = reward_fraction * self._reward_scale
        return self._make_observation(msg, final_step_reward, current_ticket)

