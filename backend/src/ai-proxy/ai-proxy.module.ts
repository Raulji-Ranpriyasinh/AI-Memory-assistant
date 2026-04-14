import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { AiProxyService } from './ai-proxy.service';

@Module({
  imports: [HttpModule],
  providers: [AiProxyService],
  exports: [AiProxyService],
})
export class AiProxyModule {}
