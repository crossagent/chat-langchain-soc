from langchain.chat_models import ChatOpenAI
from agents.soc_gpt_agent import SocGPT
from langchain.callbacks.base import  AsyncCallbackHandler
from langchain.callbacks.manager import  AsyncCallbackManager

# Set up of your agent

# Conversation stages - can be modified
conversation_stages = {
    "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
    "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
    "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
    "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
    "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
    "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
    "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
}

# Agent characteristics - can be modified
config = dict(
    salesperson_name="Ted Lasso",
    salesperson_role="Business Development Representative",
    company_name="Sleep Haven",
    company_business="Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers.",
    company_values="Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service.",
    conversation_purpose="find out whether they are looking to achieve better sleep via buying a premier mattress.",
    conversation_history=[
        "Hello, this is Ted Lasso from Sleep Haven. How are you doing today? <END_OF_TURN>",
        "User: I am well, howe are you?<END_OF_TURN>",
    ],
    conversation_type="call",
    conversation_stage=conversation_stages.get(
        "1",
        "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.",
    ),
)

def get_soc_chain(
    question_handler:AsyncCallbackHandler = None, stream_handler:AsyncCallbackHandler = None, tracing: bool = False
) -> SocGPT:
    """Get the chain."""
    # init llm
    steam_manager = AsyncCallbackManager([stream_handler])
    llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=steam_manager)

    soc_agent = SocGPT.from_llm(llm, verbose=False, **config)

    # init sales agent
    soc_agent.seed_agent()

    return soc_agent
