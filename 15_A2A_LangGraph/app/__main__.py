import logging
import os
import sys

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from app.agent import Agent
from app.agent_executor import GeneralAgentExecutor


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the General Agent server with A2A protocol support."""
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise MissingAPIKeyError(
                'OPENAI_API_KEY environment variable not set.'
            )

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skills = [
            AgentSkill(
                id='web_search',
                name='Web Search Tool',
                description='Search the web for current information',
                tags=['search', 'web', 'internet'],
                examples=['What are the latest news about AI?'],
            ),
            AgentSkill(
                id='arxiv_search',
                name='Academic Paper Search',
                description='Search for academic papers on arXiv',
                tags=['research', 'papers', 'academic'],
                examples=['Find recent papers on large language models'],
            ),
            AgentSkill(
                id='rag_search',
                name='Document Retrieval',
                description='Search through loaded documents for specific information',
                tags=['documents', 'rag', 'retrieval'],
                examples=['What do the policy documents say about student loans?'],
            ),
        ]
        agent_card = AgentCard(
            name='General Purpose Agent',
            description='A helpful AI assistant with web search, academic paper search, and document retrieval capabilities',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=Agent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=Agent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )


        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                        config_store=push_config_store)
        request_handler = DefaultRequestHandler(
            agent_executor=GeneralAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender= push_sender
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
