import type { Context } from '@deepseek-ai/cordis';
import type { AttachmentStore, ImageAttachmentRef } from '@deepseek-ai/dsh-attachment';
import type { AdapterRegistrationHandle, LlmConfigurableProvider } from '@deepseek-ai/dsh-llm';
import { AgyAcpAdapter } from './adapter.js';
import { Config, DEFAULT_PROVIDER_NAME, resolveAgyAcpConfig, type RawPluginConfig } from './config.js';
import type { ImageBytesLoader } from './session-sync.js';

export const name = 'dsh-agy-native';

export const inject = ['llm'] as const;

export { Config, AgyAcpAdapter, resolveAgyAcpConfig };

export function apply(ctx: Context, config: RawPluginConfig = {}): void {
  const settings = () => resolveAgyAcpConfig(config);
  const initial = settings();

  const settingsNs =
    (ctx.fiber as unknown as { entry?: { options?: { id?: string } } })?.entry?.options?.id ??
    'agy-acp';

  let lastError: string | undefined;
  let lastTierName: string | undefined;
  let adapterHandle: AdapterRegistrationHandle | undefined;
  let disposed = false;

  const makeDirectoryEntry = (error?: string): LlmConfigurableProvider => {
    const cur = settings();
    const displayName =
      lastTierName && cur.providerName === DEFAULT_PROVIDER_NAME
        ? `Antigravity (${lastTierName})`
        : cur.providerName;
    return {
      provider: initial.providerId,
      displayName,
      settingsNs,
      settingsPath: [],
      declared: false,
      nativeAuth: {
        env: [],
        oauth: true,
      },
      ...(error ? { error } : {}),
    };
  };

  const directoryHandle = ctx.llm.registerConfigurableProviders([makeDirectoryEntry()]);

  const loadImage: ImageBytesLoader = async (
    ref: ImageAttachmentRef,
    signal?: AbortSignal,
  ) => {
    const attachments = ctx.get('attachments' as never, false) as AttachmentStore | undefined;
    if (!attachments?.readImage) return undefined;
    const stored = await attachments.readImage(ref, signal);
    return {
      base64: Buffer.from(stored.data).toString('base64'),
      mimeType: stored.ref.mediaType,
    };
  };

  const adapter = new AgyAcpAdapter({
    getConfig: settings,
    loadImage,
    onCatalogError: (err) => {
      if (disposed) return;
      if (err !== lastError) {
        lastError = err;
        directoryHandle.replace([makeDirectoryEntry(err)]);
      }
    },
    onTierUpdated: (tierName) => {
      if (disposed) return;
      lastTierName = tierName;
      adapterHandle?.replace([initial.providerId]);
      directoryHandle.replace([makeDirectoryEntry(lastError)]);
    },
  });

  lastTierName = adapter.catalog.tier?.name;
  adapterHandle = ctx.llm.registerAdapter([initial.providerId], adapter);
  if (lastTierName) {
    directoryHandle.replace([makeDirectoryEntry()]);
  }

  ctx.effect(() => () => {
    disposed = true;
    void adapter.dispose();
  });
}

export default {
  name,
  inject,
  Config,
  apply,
};
