# THIS CLASS USED TO CONTAIN THE DJANGO MODELS, HOWEVER AS THEY ARE NEEDED BY THE WORKER APP
# AS WELL THEY HAVE BEEN MOVED TO A SHARED LOCATION. THE `from redbox.models.shared import` HAS BEEN ADDED TO AVOID
# BREAKING EXISTING REFERENCES.
# NEW MODELS SHOULD NOT BE ADDED TO THIS FILE

from redbox.models.shared import (  # noqa: F401
    ActivityEvent,
    Agent,
    AgentPlan,
    AgentTool,
    AISettings,
    Chat,
    ChatLLMBackend,
    ChatMessage,
    ChatMessageFeedback,
    ChatMessageTokenUse,
    Citation,
    File,
    FileTeamMembership,
    FileTool,
    MonitorSearchRoute,
    MonitorWebSearchResults,
    SSOUserManager,
    Team,
    TeamTool,
    Tool,
    ToolAccessRule,
    ToolManager,
    ToolSettings,
    User,
    UserSSO,
    UserSSOAttribute,
    UserTeamMembership,
    UserTool,
    build_s3_key,
)
