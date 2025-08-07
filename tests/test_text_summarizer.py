"""Tests for the summarizer agent functionality."""

# Standard library imports
# Standart library imports
from typing import cast
from unittest.mock import MagicMock, patch

# Thirdparty imports
# Third-party imports
from langchain_core.messages import AIMessage, HumanMessage

# Local imports
from ds_capstone.summarizer_graph import AgentState, SummarizerAgent, get_current_date


class TestGetCurrentDate:
    """Test cases for the get_current_date tool function."""

    @patch('ds_capstone.summarizer_graph.datetime')
    def test_get_current_date_format(self, mock_datetime):
        """Test that get_current_date returns correctly formatted date.

        Verifies that:
        - Function returns date in expected format
        - Uses strftime with correct pattern
        """
        # Mock datetime to return a specific date
        mock_now = MagicMock()
        mock_now.strftime.return_value = "Monday, January 15, 2024"
        mock_datetime.now.return_value = mock_now

        result = get_current_date.invoke({})  # Use invoke method for tool

        assert result == "Monday, January 15, 2024"
        mock_datetime.now.assert_called_once()
        mock_now.strftime.assert_called_once_with("%A, %B %d, %Y")

    def test_get_current_date_returns_string(self):
        """Test that get_current_date returns a string."""
        result = get_current_date.invoke({})  # Use invoke method for tool
        assert isinstance(result, str)
        assert len(result) > 0


class TestSummarizerAgent:
    """Test cases for the SummarizerAgent class."""

    @patch('ds_capstone.summarizer_graph.ChatOllama')
    def test_init(self, mock_chat_ollama):
        """Test initialization of SummarizerAgent.

        Verifies that:
        - Tools are set up correctly
        - LLM is initialized with correct model and temperature
        - Graph is compiled
        """
        mock_llm = MagicMock()
        mock_model_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_model_with_tools
        mock_chat_ollama.return_value = mock_llm

        agent = SummarizerAgent(model_name="test-model")

        # Verify initialization
        assert len(agent.tools) == 1  # Should have get_current_date tool
        assert agent.llm == mock_llm
        assert agent.model_with_tools == mock_model_with_tools
        assert hasattr(agent, 'graph')

        # Verify LLM was initialized with correct parameters
        mock_chat_ollama.assert_called_once()
        call_kwargs = mock_chat_ollama.call_args[1]
        assert call_kwargs['model'] == "test-model"

    @patch('ds_capstone.summarizer_graph.ChatOllama')
    def test_tool_router_with_tool_calls(self, mock_chat_ollama):
        """Test tool router when message has tool calls.

        Verifies that:
        - Router returns "tools" when tool calls are present
        """
        mock_llm = MagicMock()
        mock_chat_ollama.return_value = mock_llm

        agent = SummarizerAgent()

        # Create a mock message with tool calls
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "get_current_date", "args": {}}]

        state = cast(AgentState, {"messages": [mock_message]})
        result = agent.tool_router(state)

        assert result == "tools"

    @patch('ds_capstone.summarizer_graph.ChatOllama')
    def test_tool_router_without_tool_calls(self, mock_chat_ollama):
        """Test tool router when message has no tool calls.

        Verifies that:
        - Router returns END when no tool calls are present
        """
        mock_llm = MagicMock()
        mock_chat_ollama.return_value = mock_llm

        agent = SummarizerAgent()

        # Create a mock message without tool calls
        mock_message = AIMessage(content="This is a regular response")

        state = cast(AgentState, {"messages": [mock_message]})
        result = agent.tool_router(state)

        assert result == "__end__"

    @patch('ds_capstone.summarizer_graph.ChatOllama')
    def test_sum_agent(self, mock_chat_ollama):
        """Test the main agent processing function.

        Verifies that:
        - Prompt template is created with system prompt
        - Chain is invoked with message history
        - Response is added to messages
        """
        mock_llm = MagicMock()
        mock_model_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_model_with_tools
        mock_chat_ollama.return_value = mock_llm

        # Mock the chain result
        mock_response = AIMessage(content="Test response")

        with patch('ds_capstone.summarizer_graph.ChatPromptTemplate') as mock_prompt_template:
            mock_template = MagicMock()
            mock_prompt_template.from_messages.return_value = mock_template

            mock_chain = MagicMock()
            mock_template.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.invoke.return_value = mock_response

            agent = SummarizerAgent()

            # Test input state
            input_state = cast(AgentState, {"messages": [HumanMessage(content="Test message")]})

            _ = agent.sum_agent(input_state)

            # Verify chain was invoked with correct parameters
            mock_chain.invoke.assert_called_once()
            call_args = mock_chain.invoke.call_args[0][0]
            assert "history" in call_args
            assert call_args["history"] == input_state["messages"]

    @patch('ds_capstone.summarizer_graph.ChatOllama')
    def test_execute_simple_message(self, mock_chat_ollama):
        """Test executing a simple message through the agent.

        Verifies that:
        - Graph processes the input message
        - Returns appropriate response
        """
        mock_llm = MagicMock()
        mock_model_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_model_with_tools
        mock_chat_ollama.return_value = mock_llm

        agent = SummarizerAgent()

        # Mock the graph invoke method
        mock_final_state = {"messages": [HumanMessage(content="Test input"), AIMessage(content="Test response")]}
        agent.graph.invoke = MagicMock(return_value=mock_final_state)

        result = agent.execute("Test input")

        # Verify graph was called with correct initial state
        agent.graph.invoke.assert_called_once()
        call_args = agent.graph.invoke.call_args[0][0]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0].content == "Test input"

        # Verify result extraction
        assert result == "Test response"

    @patch('ds_capstone.summarizer_graph.ChatOllama')
    def test_execute_with_tool_usage(self, mock_chat_ollama):
        """Test executing a message that triggers tool usage."""
        mock_llm = MagicMock()
        mock_model_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_model_with_tools
        mock_chat_ollama.return_value = mock_llm

        agent = SummarizerAgent()

        # Mock a conversation that involves tool usage
        mock_final_state = {
            "messages": [
                HumanMessage(content="What is today's date?"),
                AIMessage(content="I'll check the current date for you."),
                AIMessage(content="Today is Monday, January 15, 2024"),
            ]
        }
        agent.graph.invoke = MagicMock(return_value=mock_final_state)

        result = agent.execute("What is today's date?")

        # Should return the final AI message
        assert "Monday, January 15, 2024" in result

    def test_tools_list_contains_get_current_date(self):
        """Test that the agent's tools list includes the get_current_date tool."""
        with patch('ds_capstone.summarizer_graph.ChatOllama'):
            agent = SummarizerAgent()

            # Verify that get_current_date is in the tools
            tool_names = [tool.name for tool in agent.tools]
            assert "get_current_date" in tool_names
