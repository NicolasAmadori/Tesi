import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.runnables import RunnableConfig

examples = [
    {
        "text": (
            "The PhD program in Computer Science and Engineering lasts three years "
            "and offers a broad-spectrum preparation in both scientific and "
            "engineering aspects of computing."
        ),
        "head": "PhD program in Computer Science and Engineering",
        "head_type": "Program",
        "relation": "HAS_DURATION",
        "tail": "three years",
        "tail_type": "Duration",
    },
    {
        "text": (
            "The PhD program in Computer Science and Engineering lasts three years "
            "and offers a broad-spectrum preparation in both scientific and "
            "engineering aspects of computing."
        ),
        "head": "PhD program in Computer Science and Engineering",
        "head_type": "Program",
        "relation": "OFFERS",
        "tail": "broad-spectrum preparation",
        "tail_type": "Offering",
    },
    {
        "text": (
            "At the beginning of the PhD program, each student is assigned a supervisor "
            "plus a co-supervisor who guides the student throughout the Ph.D. studies."
        ),
        "head": "PhD program",
        "head_type": "Program",
        "relation": "REQUIRES",
        "tail": "supervisor",
        "tail_type": "Role",
    },
    {
        "text": (
            "The learning plan foresees reaching 30 CFU (credits) by attending courses "
            "and passing evaluations."
        ),
        "head": "learning plan",
        "head_type": "Requirement",
        "relation": "HAS_ATTRIBUTE",
        "tail": "30 CFU",
        "tail_type": "Credit",
    },
    {
        "text": (
            "The Course is held in Italian. The Computer Science and Engineering "
            "program is offered by the University of Bologna."
        ),
        "head": "Computer Science and Engineering program",
        "head_type": "Program",
        "relation": "OFFERED_BY",
        "tail": "University of Bologna",
        "tail_type": "Institution",
    },
    {
        "text": (
            "Le lezioni del corso di Laurea in Ingegneria e Scienze Informatiche "
            "si svolgeranno in presenza nell'a.a. 2024/25."
        ),
        "head": "Lezioni del corso di Laurea in Ingegneria e Scienze Informatiche",
        "head_type": "Event",
        "relation": "HAS_DELIVERY_METHOD",
        "tail": "in presenza",
        "tail_type": "DeliveryMethod",
    },
    {
        "text": (
            "Il corso di Architetture degli Elaboratori è suddiviso in due moduli, "
            "sia per la classe A che per la classe B."
        ),
        "head": "corso di Architetture degli Elaboratori",
        "head_type": "Course",
        "relation": "IS_DIVIDED_INTO",
        "tail": "due moduli",
        "tail_type": "Module",
    },
    {
        "text": (
            "Il periodo di lezione del corso Electromagnetic Propagation for Wireless Systems "
            "si estende da settembre 2017 fino al 18 dicembre 2017."
        ),
        "head": "corso Electromagnetic Propagation for Wireless Systems",
        "head_type": "Course",
        "relation": "HAS_DURATION",
        "tail": "settembre 2017 - 18 dicembre 2017",
        "tail_type": "TimePeriod",
    },
    {
        "text": (
            "Il Consiglio di Corso si avvale di commissioni costituite da docenti del corso "
            "per svolgere alcune delle sue funzioni."
        ),
        "head": "Consiglio di Corso",
        "head_type": "OrganizationalBody",
        "relation": "IS_ASSISTED_BY",
        "tail": "commissioni costituite da docenti del corso",
        "tail_type": "OrganizationalUnit",
    },
    {
        "text": (
            "Il corso di Laurea in Ingegneria e Scienze Informatiche è offerto dall'Università di Bologna."
        ),
        "head": "corso di Laurea in Ingegneria e Scienze Informatiche",
        "head_type": "DegreeProgram",
        "relation": "OFFERED_BY",
        "tail": "Università di Bologna",
        "tail_type": "Institution",
    },
    {
        "text": "Consulta il Calendario Didattico delle lezioni del Corso di Studio.",
        "head": "Corso di Studio",
        "head_type": "Program",
        "relation": "HAS_SCHEDULE",
        "tail": "Calendario Didattico",
        "tail_type": "Schedule",
    },
    {
        "text": "ALGEBRA E GEOMETRIA / (1) Modulo 1",
        "head": "ALGEBRA E GEOMETRIA",
        "head_type": "Course",
        "relation": "HAS_PART",
        "tail": "Modulo 1",
        "tail_type": "CourseModule",
    },
    {
        "text": "Crediti formativi: 9",
        "head": "Corso",
        "head_type": "Course",
        "relation": "HAS_CREDITS",
        "tail": "9",
        "tail_type": "CreditValue",
    },
    {
        "text": "Periodo di lezione: September 2017 - 18 December 2017",
        "head": "Periodo di lezione",
        "head_type": "AcademicPeriod",
        "relation": "HAS_DURATION",
        "tail": "September 2017 - 18 December 2017",
        "tail_type": "DateRange",
    },
    {
        "text": "Docente: ORESTE ANDRISANO",
        "head": "ORESTE ANDRISANO",
        "head_type": "Person",
        "relation": "HAS_ROLE",
        "tail": "Docente",
        "tail_type": "Role",
    },
    {
        "text": "Luogo: AULA 4.1 - Piano Terra - Viale del Risorgimento 2 - Bologna",
        "head": "AULA 4.1",
        "head_type": "Location",
        "relation": "LOCATED_AT",
        "tail": "Piano Terra - Viale del Risorgimento 2 - Bologna",
        "tail_type": "Address",
    },
    {
        "text": "Iscriversi al Corso: requisiti, tempi e modalità",
        "head": "Corso",
        "head_type": "Program",
        "relation": "HAS_ENROLLMENT_INFO",
        "tail": "requisiti, tempi e modalità",
        "tail_type": "EnrollmentDetails",
    },
    {
        "text": "Tasse: importi ed esoneri",
        "head": "Tasse",
        "head_type": "Fee",
        "relation": "HAS_DETAILS",
        "tail": "importi ed esoneri",
        "tail_type": "FeeDetails",
    },
    {
        "text": "Iscriversi ad anni successivi al primo",
        "head": "Iscrizione",
        "head_type": "Process",
        "relation": "APPLIES_TO",
        "tail": "anni successivi al primo",
        "tail_type": "AcademicYear",
    },

]

system_prompt = (
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph from various types of text, including "
    "educational program descriptions, institutional information, and general facts.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities, concepts, and attributes.\n"
    "- **Relationships** represent connections between nodes.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Use general types for node labels such as 'Program', 'Degree', "
    "'Institution', 'Duration', 'Requirement', 'Person', etc.\n"
    "- **Node IDs**: Use names or human-readable identifiers found in the text. Avoid "
    "using integers as node IDs.\n"
    "## 3. Relationships\n"
    "- Use general and timeless relationship types.\n"
    "- Examples: 'HAS_DURATION', 'REQUIRES', 'OFFERS', 'PART_OF', 'LEADS_TO', etc.\n"
    "## 4. Attributes\n"
    "- When appropriate, use attributes to add details to nodes instead of creating "
    "new nodes for every piece of information.\n"
    "## 5. Coreference Resolution\n"
    "- Maintain entity consistency throughout the graph.\n"
    "## 6. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)


def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Person'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "Instead of using specific and momentary types such as "
            "'BECAME_PROFESSOR', use more general and timeless relationship types "
            "like 'PROFESSOR'. However, do not sacrifice any accuracy for generality"
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    llm_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    # Only openai supports enum param
    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=enum_values,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "Below are a number of examples of text and their extracted "
        "entities and relationships."
        "{examples}\n"
        "For the following text, extract entities and relations as "
        "in the provided example."
        "IMPORTANT NOTE: You are NOT forced to use just the entity types and relationships that you find in the examples,"
        " instead, it will be very good if you are able to generate new relation and entity types"
        "{format_instructions}\nText: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of source node",
            ),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of target node",
            ),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                description="The type of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                "The relationship property 'id' is reserved and cannot be used."
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            Field(None, description="List of relationship properties"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph


def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value
    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )


def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        node_properties = {}
        if "properties" in node and node["properties"]:
            for p in node["properties"]:
                node_properties[format_property_key(p["key"])] = p["value"]
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type"),
                properties=node_properties,
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        rel_properties = {}
        if "properties" in rel and rel["properties"]:
            for p in rel["properties"]:
                rel_properties[format_property_key(p["key"])] = p["value"]

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
                properties=rel_properties,
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else None,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["function_call"]["arguments"]
                )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM.

    It allows specifying constraints on the types of nodes and relationships to include
    in the output graph. The class supports extracting properties for both nodes and
    relationships.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to an empty list, allowing all relationship
          types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to
          the LLM with additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply
          filtering to strictly adhere to `allowed_nodes` and `allowed_relationships`.
          Defaults to True.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any
          node properties from text. Alternatively, a list of valid properties can
          be provided for the LLM to extract, restricting extraction to those specified.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract
          any relationship properties from text. Alternatively, a list of valid
          properties can be provided for the LLM to extract, restricting extraction to
          those specified.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm=ChatOpenAI(temperature=0)
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization"])

            doc = Document(page_content="Elon Musk is suing OpenAI")
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = True
        # Check if the LLM really supports structured output
        try:
            llm.with_structured_output(_Graph)
        except NotImplementedError:
            self._function_call = False
        if not self._function_call:
            if node_properties or relationship_properties:
                raise ValueError(
                    "The 'node_properties' and 'relationship_properties' parameters "
                    "cannot be used in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair  # type: ignore

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.chain = prompt | llm
        else:
            # Define chain
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes,
                allowed_relationships,
                node_properties,
                llm_type,
                relationship_properties,
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or default_prompt
            self.chain = prompt | structured_llm

    def process_response(self, document: Document) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        raw_schema = self.chain.invoke({"input": text})
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            nodes_set = set()
            relationships = []
            if not isinstance(raw_schema, str):
                raw_schema = raw_schema.content
            parsed_json = self.json_repair.loads(raw_schema)
            i = 0
            for upper_rel in parsed_json:
              #if len(rel) > 0:
              #  if type(rel) == list:
              #    rel = rel[0]
              for rel in upper_rel:
                if not isinstance(rel, dict):
                  continue
                if "head" not in rel or "head_type" not in rel or "tail" not in rel or "tail_type" not in rel:
                    print(f"This relationships is not correct: {rel}")
                    continue
                # Nodes need to be deduplicated using a set
                nodes_set.add((rel["head"], rel["head_type"]))
                nodes_set.add((rel["tail"], rel["tail_type"]))

                source_node = Node(id=rel["head"], type=rel["head_type"])
                target_node = Node(id=rel["tail"], type=rel["tail_type"])
                relationships.append(
                    Relationship(
                        source=source_node, target=target_node, type=rel["relation"]
                    )
                )
            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document) for document in documents]

    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a
        graph document.
        """
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text}, config=config)
        raw_schema = cast(Dict[Any, Any], raw_schema)
        nodes, relationships = _convert_to_graph_document(raw_schema)

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results