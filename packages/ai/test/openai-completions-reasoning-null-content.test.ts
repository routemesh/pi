import { Type } from "typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getModel } from "../src/models.ts";
import { streamSimple } from "../src/stream.ts";
import type { AssistantMessage, Model, Tool, ToolResultMessage, UserMessage } from "../src/types.ts";

// streamSimple returns { result: () => Promise<AssistantMessage> }
// The result() promise resolves to the final AssistantMessage

const mockState = vi.hoisted(() => ({
	lastParams: undefined as unknown,
	chunks: undefined as
		| Array<null | {
				id?: string;
				choices?: Array<{ delta: Record<string, unknown>; finish_reason: string | null; usage?: unknown }>;
				usage?: {
					prompt_tokens: number;
					completion_tokens: number;
					prompt_tokens_details: { cached_tokens: number; cache_write_tokens?: number };
					completion_tokens_details: { reasoning_tokens: number };
				};
		  }>
		| undefined,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (params: unknown) => {
					mockState.lastParams = params;
					const stream = {
						async *[Symbol.asyncIterator]() {
							const chunks = mockState.chunks ?? [
								{
									choices: [{ delta: {}, finish_reason: "stop" }],
									usage: {
										prompt_tokens: 1,
										completion_tokens: 1,
										prompt_tokens_details: { cached_tokens: 0 },
										completion_tokens_details: { reasoning_tokens: 0 },
									},
								},
							];
							for (const chunk of chunks) {
								yield chunk;
							}
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}

	return { default: FakeOpenAI };
});

describe("openai-completions: reasoning_content without content (issue #4909)", () => {
	beforeEach(() => {
		mockState.lastParams = undefined;
		mockState.chunks = undefined;
	});

	it("handles model returning reasoning_content + tool_calls but no content", async () => {
		// Simulate a reasoning model (e.g. GLM-5.2 on Fireworks) that returns:
		// - reasoning_content (thinking) via delta
		// - tool_calls via delta
		// - NO content field at all
		// This causes message.content to be null/empty, which crashes with
		// "content is not iterable" when the agent loop tries to process it.
		mockState.chunks = [
			// First chunk: role
			{
				choices: [{ delta: { role: "assistant" }, finish_reason: null }],
			},
			// Reasoning content (thinking)
			{
				choices: [{ delta: { reasoning_content: "I need to call the read tool" }, finish_reason: null }],
			},
			// Tool call — NO content field in this delta
			{
				choices: [{
					delta: {
						tool_calls: [{
							index: 0,
							id: "call_001",
							type: "function",
							function: { name: "read", arguments: '{"path":"/etc/hostname"}' },
						}],
					},
					finish_reason: null,
				}],
			},
			// Final chunk: finish_reason = tool_calls
			{
				choices: [{ delta: {}, finish_reason: "tool_calls" }],
				usage: {
					prompt_tokens: 100,
					completion_tokens: 50,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 30 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = {
			...baseModel,
			api: "openai-completions",
			reasoning: true,
			compat: {
				supportsReasoningEffort: false,
				supportsDeveloperRole: false,
			},
		} as const;

		const tools: Tool[] = [
			{
				name: "read",
				description: "Read a file",
				parameters: Type.Object({
					path: Type.String(),
				}),
			},
		];

		const userMessage: UserMessage = {
			role: "user",
			content: [{ type: "text", text: "Read /etc/hostname and tell me what it says" }],
		};

		const streamResult = streamSimple(
			model as Model<"openai-completions">,
			{
				systemPrompt: "You are a helpful assistant.",
				messages: [userMessage],
				tools,
			},
			{
				apiKey: "test-key",
				thinkingLevel: "medium",
			},
		);

		// Collect the stream output
		const message = await streamResult.result();

		// The stream should NOT crash with "content is not iterable"
		// If it crashes, the promise will reject and the test will fail
		expect(message).toBeDefined();
		expect(message.stopReason).toBe("toolUse");
		
		// Content should be an array (possibly empty, but NOT null/undefined)
		expect(Array.isArray(message.content)).toBe(true);
		
		// Should have a tool call block
		const toolCalls = message.content.filter((c: any) => c.type === "toolCall");
		expect(toolCalls.length).toBe(1);
		expect(toolCalls[0].name).toBe("read");
	});

	it("handles multi-turn: tool result followed by model response with reasoning only", async () => {
		// Second turn: model returns reasoning_content and text after tool result
		// The assistant message from the first turn has content: null (no text, only tool calls)
		// This tests that convertMessages handles null content on the previous assistant message
		mockState.chunks = [
			{
				choices: [{ delta: { role: "assistant" }, finish_reason: null }],
			},
			{
				choices: [{ delta: { reasoning_content: "The file contains the hostname" }, finish_reason: null }],
			},
			{
				choices: [{ delta: { content: "The hostname is my-machine" }, finish_reason: null }],
			},
			{
				choices: [{ delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 200,
					completion_tokens: 50,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 20 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = {
			...baseModel,
			api: "openai-completions",
			reasoning: true,
			compat: {
				supportsReasoningEffort: false,
				supportsDeveloperRole: false,
			},
		} as const;

		const tools: Tool[] = [
			{
				name: "read",
				description: "Read a file",
				parameters: Type.Object({
					path: Type.String(),
				}),
			},
		];

		// Messages: user, assistant (with tool call, content: null), tool result
		const userMessage: UserMessage = {
			role: "user",
			content: [{ type: "text", text: "Read /etc/hostname" }],
		};
		const assistantMessage: AssistantMessage = {
			role: "assistant",
			content: [
				{ type: "toolCall", id: "call_001", name: "read", arguments: { path: "/etc/hostname" } },
			],
			stopReason: "toolUse",
		};
		const toolResultMessage: ToolResultMessage = {
			role: "toolResult",
			content: [
				{ type: "toolResult", toolCallId: "call_001", toolName: "read", result: "my-machine" },
			],
		};

		const streamResult = streamSimple(
			model as Model<"openai-completions">,
			{
				systemPrompt: "You are a helpful assistant.",
				messages: [userMessage, assistantMessage, toolResultMessage],
				tools,
			},
			{
				apiKey: "test-key",
				thinkingLevel: "medium",
			},
		);

		const message = await streamResult.result();

		// Should NOT crash with "content is not iterable"
		expect(message).toBeDefined();
		expect(message.stopReason).toBe("stop");
		expect(Array.isArray(message.content)).toBe(true);
		
		// Should have text content
		const textBlocks = message.content.filter((c: any) => c.type === "text");
		expect(textBlocks.length).toBeGreaterThan(0);
	});
});
