import { config } from "./config";

export interface VeldTodo {
  id: string;
  content: string;
  status: "backlog" | "todo" | "in_progress" | "blocked" | "done" | "cancelled";
  priority: "urgent" | "high" | "medium" | "low" | "none";
  due_date?: string;
  project?: string;
  tags: string[];
  external_id?: string;
}

export interface VeldProject {
  id: string;
  name: string;
  prefix?: string;
  description?: string;
}

async function apiCall<T>(
  endpoint: string,
  method: string = "POST",
  body?: object
): Promise<T> {
  const url = `${config.veld.apiUrl}${endpoint}`;

  const res = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": config.veld.apiKey,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Veld API error: ${res.status} ${text}`);
  }

  return res.json();
}

export async function listTodos(includeCompleted = false): Promise<VeldTodo[]> {
  const statuses = includeCompleted
    ? ["backlog", "todo", "in_progress", "blocked", "done"]
    : ["backlog", "todo", "in_progress", "blocked"];

  const data = await apiCall<{ todos: VeldTodo[] }>("/api/todos/list", "POST", {
    user_id: config.veld.userId,
    status: statuses,
  });
  return data.todos || [];
}

export async function listProjects(): Promise<VeldProject[]> {
  const data = await apiCall<{ projects: [VeldProject, unknown][] | VeldProject[] }>("/api/projects/list", "POST", {
    user_id: config.veld.userId,
  });

  // Handle both tuple format [project, stats] and flat format
  const projects = data.projects || [];
  return projects.map((p: any) => Array.isArray(p) ? p[0] : p);
}

export async function createProject(name: string, prefix?: string): Promise<VeldProject> {
  const data = await apiCall<{ project: VeldProject }>("/api/projects", "POST", {
    user_id: config.veld.userId,
    name,
    prefix,
  });
  return data.project;
}

export async function createTodo(todo: {
  content: string;
  priority?: string;
  due_date?: string;
  project?: string;
  tags?: string[];
  external_id?: string;
}): Promise<VeldTodo> {
  const data = await apiCall<{ todo: VeldTodo }>("/api/todos", "POST", {
    user_id: config.veld.userId,
    ...todo,
  });
  return data.todo;
}

export async function updateTodo(
  todoId: string,
  updates: Partial<VeldTodo>
): Promise<void> {
  await apiCall(`/api/todos/${todoId}/update`, "POST", {
    user_id: config.veld.userId,
    ...updates,
  });
}

export async function completeTodo(todoId: string): Promise<void> {
  await apiCall(`/api/todos/${todoId}/complete`, "POST", {
    user_id: config.veld.userId,
  });
}

export async function findByExternalId(externalId: string): Promise<VeldTodo | null> {
  const todos = await listTodos(true);
  return todos.find((t) => t.external_id === externalId) || null;
}
