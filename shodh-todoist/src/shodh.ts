import { config } from "./config";

export interface ShodhTodo {
  id: string;
  content: string;
  status: "backlog" | "todo" | "in_progress" | "blocked" | "done" | "cancelled";
  priority: "urgent" | "high" | "medium" | "low" | "none";
  due_date?: string;
  project?: string;
  tags: string[];
  external_id?: string;
}

export interface ShodhProject {
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
  const url = `${config.shodh.apiUrl}${endpoint}`;

  const res = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": config.shodh.apiKey,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Shodh API error: ${res.status} ${text}`);
  }

  return res.json();
}

export async function listTodos(includeCompleted = false): Promise<ShodhTodo[]> {
  const statuses = includeCompleted
    ? ["backlog", "todo", "in_progress", "blocked", "done"]
    : ["backlog", "todo", "in_progress", "blocked"];

  const data = await apiCall<{ todos: ShodhTodo[] }>("/api/todos/list", "POST", {
    user_id: config.shodh.userId,
    status: statuses,
  });
  return data.todos || [];
}

export async function listProjects(): Promise<ShodhProject[]> {
  const data = await apiCall<{ projects: [ShodhProject, unknown][] | ShodhProject[] }>("/api/projects/list", "POST", {
    user_id: config.shodh.userId,
  });

  // Handle both tuple format [project, stats] and flat format
  const projects = data.projects || [];
  return projects.map((p: any) => Array.isArray(p) ? p[0] : p);
}

export async function createProject(name: string, prefix?: string): Promise<ShodhProject> {
  const data = await apiCall<{ project: ShodhProject }>("/api/projects", "POST", {
    user_id: config.shodh.userId,
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
}): Promise<ShodhTodo> {
  const data = await apiCall<{ todo: ShodhTodo }>("/api/todos", "POST", {
    user_id: config.shodh.userId,
    ...todo,
  });
  return data.todo;
}

export async function updateTodo(
  todoId: string,
  updates: Partial<ShodhTodo>
): Promise<void> {
  await apiCall(`/api/todos/${todoId}/update`, "POST", {
    user_id: config.shodh.userId,
    ...updates,
  });
}

export async function completeTodo(todoId: string): Promise<void> {
  await apiCall(`/api/todos/${todoId}/complete`, "POST", {
    user_id: config.shodh.userId,
  });
}

export async function findByExternalId(externalId: string): Promise<ShodhTodo | null> {
  const todos = await listTodos(true);
  return todos.find((t) => t.external_id === externalId) || null;
}
