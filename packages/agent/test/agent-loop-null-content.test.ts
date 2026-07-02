import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { agentLoop } from "../src/agent-loop.ts";
import type { AgentContext, AgentLoopConfig, AgentMessage, AgentTool } from "../src/types.ts";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createModel(): Model<"openai-completions"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-completions",
		provider: "mock",
		baseUrl: "https://example.invalid",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
}

function createTool(): AgentTool {
	return {
		name: "read",
		description: "Read a file",
		parameters: Type.Object({ path: Type.String() }),
		execute: async () => ({ text: "file contents here" }),
	};
}

describe("agent-loop: null content with tool calls (issue #4909)", () => {
	it("does not crash when assistant message has content: null and stopReason: toolUse", async () => {
		// Reproduces the "content is not iterable" crash.
		// When a reasoning model (e.g. GLM-5.2 on Fireworks) returns reasoning_content
		// and tool_calls but no text content, the AssistantMessage.content can be null.
		// The agent loop calls message.content.filter() which throws.
		const model = createModel();
		const tool = createTool();

		const config: AgentLoopConfig = {
			model,
			convertToLlm: identityConverter,
		};

		const context: AgentContext = {
			model,
			messages: [],
			tools: [tool],
			systemPrompt: "You are a helpful assistant.",
		};

		const userMessage: UserMessage = {
			role: "user",
			content: [{ type: "text", text: "Read /etc/hostname" }],
		};

		// Mock stream: first call returns a message with content: null + toolUse
		// second call returns a normal stop response
		let callCount = 0;
		const streamFn = () => {
			callCount++;
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callCount === 1) {
					// The model returned reasoning_content + tool_calls but no content
					// This results in content: null on the AssistantMessage
					stream.push({
						type: "done",
						reason: "toolUse",
						message: {
							role: "assistant",
							content: null as unknown as AssistantMessage["content"],
							stopReason: "toolUse",
						} as AssistantMessage,
					});
				} else {
					stream.push({
						type: "done",
						reason: "stop",
						message: {
							role: "assistant",
							content: [{ type: "text", text: "The hostname is my-machine" }],
							stopReason: "stop",
						} as AssistantMessage,
					});
				}
			});
			return stream;
		};

		const events: any[] = [];
		const eventStream = agentLoop([userMessage] as AgentMessage[], context, config, undefined, streamFn as any);

		// This should NOT throw "content is not iterable"
		try {
			for await (const event of eventStream) {
				events.push(event);
				if (event.type === "agent_end") break;
				// Prevent infinite loops
				if (events.length > 100) break;
			}
		} catch (e: any) {
			expect(e.message).not.toContain("content is not iterable");
			throw e;
		}

		// Should complete successfully
		expect(events.some((e) => e.type === "agent_end")).toBe(true);
	});

	it("does not crash when assistant message has content: null and stopReason: stop", async () => {
		// Edge case: model returns only reasoning, no text, no tool calls, stopReason: stop
		const model = createModel();
		const tool = createTool();

		const config: AgentLoopConfig = {
			model,
			convertToLlm: identityConverter,
		};

		const context: AgentContext = {
			model,
			messages: [],
			tools: [tool],
			systemPrompt: "You are a helpful assistant.",
		};

		const userMessage: UserMessage = {
			role: "user",
			content: [{ type: "text", text: "Hello" }],
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				stream.push({
					type: "done",
					reason: "stop",
					message: {
						role: "assistant",
						content: null as unknown as AssistantMessage["content"],
						stopReason: "stop",
					} as AssistantMessage,
				});
			});
			return stream;
		};

		const events: any[] = [];
		const eventStream = agentLoop([userMessage] as AgentMessage[], context, config, undefined, streamFn as any);

		try {
			for await (const event of eventStream) {
				events.push(event);
				if (event.type === "agent_end") break;
				if (events.length > 100) break;
			}
		} catch (e: any) {
			expect(e.message).not.toContain("content is not iterable");
			throw e;
		}

		expect(events.some((e) => e.type === "agent_end")).toBe(true);
	});
});
