import re
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod


class PromptValidationError(Exception):
    """Raised when prompt validation fails"""
    pass


class ConditionalPrompt:
    """Enhanced prompt with conditional logic support"""
    
    def __init__(self, prompt: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize ConditionalPrompt with support for conditional expressions.
        
        Syntax:
        - {if condition}content{/if}
        - {if condition}content{else}alternative{/if}
        - Standard variables: {variable_name}
        
        :param prompt: Template string with conditional logic
        :param strict: If True, raises error when required variables are missing
        :param defaults: Default values for template variables
        """
        self.prompt = prompt
        self.strict = strict
        self.defaults = defaults or {}
        self._var_pattern = re.compile(r'\{([^{}]+)\}')
        self._conditional_pattern = re.compile(r'\{if\s+([^}]+)\}(.*?)(?:\{else\}(.*?))?\{/if\}', re.DOTALL)
        
    def format_prompt(self, **kwargs) -> str:
        """Format prompt with conditional logic evaluation"""
        merged_kwargs = {**self.defaults, **kwargs}
        
        # Process conditional statements
        result = self._process_conditionals(self.prompt, merged_kwargs)
        
        # Process regular variables
        variables = self._var_pattern.findall(result)
        
        if self.strict:
            missing_vars = set(variables) - set(merged_kwargs.keys())
            if missing_vars:
                raise PromptValidationError(f"Missing required variables: {missing_vars}")
        
        # Format remaining variables
        for var in variables:
            value = merged_kwargs.get(var, "")
            result = result.replace(f"{{{var}}}", str(value))
            
        return result
    
    def _process_conditionals(self, text: str, context: Dict[str, Any]) -> str:
        """Process conditional statements in the text"""
        def replace_conditional(match):
            condition = match.group(1).strip()
            true_content = match.group(2).strip()
            false_content = match.group(3).strip() if match.group(3) else ""
            
            # Evaluate condition
            try:
                # Simple evaluation - check if variable exists and is truthy
                if condition in context:
                    condition_result = bool(context[condition])
                else:
                    # Try to evaluate as a simple expression
                    condition_result = self._evaluate_condition(condition, context)
                
                return true_content if condition_result else false_content
            except Exception:
                return false_content
        
        return self._conditional_pattern.sub(replace_conditional, text)
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate simple conditions like 'var > 5' or 'var == "value"'"""
        # Simple equality check
        if '==' in condition:
            parts = condition.split('==')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip().strip('"').strip("'")
                return str(context.get(left, "")) == right
        
        # Simple comparison
        for op in ['>', '<', '>=', '<=', '!=']:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    try:
                        left_val = float(context.get(left, 0))
                        right_val = float(right)
                        if op == '>': return left_val > right_val
                        elif op == '<': return left_val < right_val
                        elif op == '>=': return left_val >= right_val
                        elif op == '<=': return left_val <= right_val
                        elif op == '!=': return left_val != right_val
                    except (ValueError, TypeError):
                        return False
        
        # Default: check if variable exists and is truthy
        return bool(context.get(condition, False))


class BasePrompt:
    def __init__(self, prompt: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None):
        """
        Initializes the BasePrompt object with a prompt template.

        :param prompt: A string that can contain placeholders within curly braces
        :param strict: If True, raises error when required variables are missing
        :param defaults: Default values for template variables
        """
        self.prompt = prompt
        self.strict = strict
        self.defaults = defaults or {}
        self._pattern = re.compile(r"\{([^}]+)\}")
        self._validate_template()

    def _validate_template(self) -> None:
        """Validates the template syntax"""
        try:
            test_vars = {var: "test" for var in self.get_input_variables()}
            self.prompt.format(**test_vars)
        except (KeyError, ValueError) as e:
            raise PromptValidationError(f"Invalid template syntax: {e}")

    def format_prompt(self, **kwargs) -> str:
        """
        Formats the prompt string using the keyword arguments provided.

        :param kwargs: The values to substitute into the prompt string
        :return: The formatted prompt string
        :raises PromptValidationError: If strict mode and required variables are missing
        """
        variables = self.get_input_variables()
        merged_kwargs = {**self.defaults, **kwargs}
        
        if self.strict:
            missing_vars = set(variables) - set(merged_kwargs.keys())
            if missing_vars:
                raise PromptValidationError(f"Missing required variables: {missing_vars}")
        
        # Use defaults for missing variables
        format_dict = {var: merged_kwargs.get(var, self.defaults.get(var, "")) for var in variables}
        
        try:
            return self.prompt.format(**format_dict)
        except (KeyError, ValueError) as e:
            raise PromptValidationError(f"Error formatting prompt: {e}")

    def get_input_variables(self) -> List[str]:
        """
        Gets the list of input variable names from the prompt string.

        :return: List of input variable names
        """
        return self._pattern.findall(self.prompt)
    
    def validate_inputs(self, **kwargs) -> Dict[str, List[str]]:
        """
        Validates input variables and returns validation results.
        
        :param kwargs: Variables to validate
        :return: Dict with 'missing' and 'extra' keys containing respective variable names
        """
        required_vars = set(self.get_input_variables())
        provided_vars = set(kwargs.keys())
        
        return {
            'missing': list(required_vars - provided_vars),
            'extra': list(provided_vars - required_vars)
        }


class RolePrompt(BasePrompt):
    VALID_ROLES = {"system", "user", "assistant"}
    
    def __init__(self, prompt: str, role: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None):
        """
        Initializes the RolePrompt object with a prompt template and a role.

        :param prompt: A string that can contain placeholders within curly braces
        :param role: The role for the message ('system', 'user', or 'assistant')
        :param strict: If True, raises error when required variables are missing
        :param defaults: Default values for template variables
        :raises ValueError: If role is not valid
        """
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {self.VALID_ROLES}")
        
        super().__init__(prompt, strict=strict, defaults=defaults)
        self.role = role

    def create_message(self, format: bool = True, **kwargs) -> Dict[str, str]:
        """
        Creates a message dictionary with a role and a formatted message.

        :param format: Whether to format the prompt with variables
        :param kwargs: The values to substitute into the prompt string
        :return: Dictionary containing the role and the formatted message
        """
        if format:
            return {"role": self.role, "content": self.format_prompt(**kwargs)}
        
        return {"role": self.role, "content": self.prompt}


class SystemRolePrompt(RolePrompt):
    def __init__(self, prompt: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(prompt, "system", strict=strict, defaults=defaults)


class UserRolePrompt(RolePrompt):
    def __init__(self, prompt: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(prompt, "user", strict=strict, defaults=defaults)


class AssistantRolePrompt(RolePrompt):
    def __init__(self, prompt: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(prompt, "assistant", strict=strict, defaults=defaults)


class PromptTemplate(BasePrompt):
    """
    Enhanced prompt template with support for composition and advanced features.
    """
    
    def __init__(self, prompt: str, strict: bool = False, defaults: Optional[Dict[str, Any]] = None,
                 parent: Optional['PromptTemplate'] = None):
        """
        Initialize an enhanced prompt template.
        
        :param prompt: Template string
        :param strict: If True, raises error when required variables are missing
        :param defaults: Default values for template variables
        :param parent: Parent template for inheritance
        """
        super().__init__(prompt, strict=strict, defaults=defaults)
        self.parent = parent
        self._children: List['PromptTemplate'] = []
        
    def compose(self, *templates: 'PromptTemplate', separator: str = "\n\n") -> 'PromptTemplate':
        """
        Compose multiple templates into a single template.
        
        :param templates: Templates to compose
        :param separator: String to join templates
        :return: New composed template
        """
        prompts = [self.prompt] + [t.prompt for t in templates]
        combined_prompt = separator.join(prompts)
        
        # Merge defaults
        combined_defaults = {**self.defaults}
        for template in templates:
            combined_defaults.update(template.defaults)
            
        return PromptTemplate(combined_prompt, strict=self.strict, defaults=combined_defaults)
    
    def extend(self, child_prompt: str, **kwargs) -> 'PromptTemplate':
        """
        Create a child template that extends this template.
        
        :param child_prompt: Additional prompt content
        :param kwargs: Additional parameters for the child
        :return: New child template
        """
        combined_prompt = f"{self.prompt}\n\n{child_prompt}"
        combined_defaults = {**self.defaults, **kwargs.get('defaults', {})}
        
        child = PromptTemplate(
            combined_prompt,
            strict=kwargs.get('strict', self.strict),
            defaults=combined_defaults,
            parent=self
        )
        self._children.append(child)
        return child


class MessageAdapter:
    """Adapts messages to different LLM provider formats"""
    
    @staticmethod
    def to_openai(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert to OpenAI format (already in this format)"""
        return messages
    
    @staticmethod
    def to_anthropic(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert to Anthropic Claude format"""
        # Claude expects a slightly different format
        converted = []
        for msg in messages:
            if msg['role'] == 'system':
                # Claude handles system messages differently
                converted.append({"role": "user", "content": f"System: {msg['content']}"})
            else:
                converted.append(msg)
        return converted
    
    @staticmethod
    def to_cohere(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert to Cohere format"""
        # Cohere has a different structure
        chat_history = []
        message = ""
        
        for msg in messages:
            if msg['role'] == 'user':
                if message:  # If there's a pending assistant message
                    chat_history.append({"role": "CHATBOT", "message": message})
                    message = ""
                chat_history.append({"role": "USER", "message": msg['content']})
            elif msg['role'] == 'assistant':
                message = msg['content']
            elif msg['role'] == 'system':
                # Cohere handles system messages as preamble
                preamble = msg['content']
                
        return {
            "message": chat_history[-1]["message"] if chat_history else "",
            "chat_history": chat_history[:-1] if chat_history else [],
            "preamble": preamble if 'preamble' in locals() else None
        }


if __name__ == "__main__":
    # Basic usage
    prompt = BasePrompt("Hello {name}, you are {age} years old")
    print(prompt.format_prompt(name="John", age=30))

    # With defaults
    prompt_with_defaults = BasePrompt(
        "Hello {name}, you are {age} years old",
        defaults={"age": 25}
    )
    print(prompt_with_defaults.format_prompt(name="Jane"))
    
    # Strict mode
    try:
        strict_prompt = BasePrompt("Hello {name}", strict=True)
        strict_prompt.format_prompt()  # This will raise an error
    except PromptValidationError as e:
        print(f"Validation error: {e}")
    
    # Role prompts
    system = SystemRolePrompt("You are a helpful assistant", defaults={"tone": "friendly"})
    print(system.create_message())
    
    # Conditional prompts
    conditional = ConditionalPrompt(
        "Hello {name}! {if premium}Welcome to our premium service!{else}Consider upgrading.{/if}"
    )
    print(conditional.format_prompt(name="Alice", premium=True))
    print(conditional.format_prompt(name="Bob", premium=False))
    
    # Template composition
    base_template = PromptTemplate("You are an AI assistant.")
    task_template = PromptTemplate("Your task is to {task}.")
    composed = base_template.compose(task_template)
    print(composed.format_prompt(task="help with coding"))
    
    # Message adaptation
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello!"}
    ]
    print("Anthropic format:", MessageAdapter.to_anthropic(messages))
