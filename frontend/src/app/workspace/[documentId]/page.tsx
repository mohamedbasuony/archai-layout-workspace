import { DocumentChatWorkspace } from "@/components/workspace/DocumentChatWorkspace";

interface WorkspaceDocumentPageProps {
  params: Promise<{ documentId: string }>;
}

export default async function WorkspaceDocumentPage({ params }: WorkspaceDocumentPageProps) {
  const { documentId } = await params;
  return <DocumentChatWorkspace initialDocumentId={documentId} />;
}
