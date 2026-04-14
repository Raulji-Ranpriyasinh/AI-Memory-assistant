export class TokenResponseDto {
  accessToken: string;
  user: {
    id: string;
    email: string;
    role: string;
    status: string;
    profile?: {
      firstName?: string;
      lastName?: string;
      language?: string;
      timezone?: string;
    };
  };
}
