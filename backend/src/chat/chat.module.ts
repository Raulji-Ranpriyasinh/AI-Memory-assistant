import { Module } from '@nestjs/common';
import { AiProxyModule } from '../ai-proxy/ai-proxy.module';
import { ChatController } from './chat.controller';

@Module({
  imports: [AiProxyModule],
  controllers: [ChatController],
})
export class ChatModule {}
